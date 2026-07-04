import json
import logging
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, field_validator

import config_api
from services.features.features import FeaturesService
from settings.base_settings import base_settings

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "data/models_iep/model_2026_05_04.cbm"
FEATURES_NAMES_PATH = BASE_DIR / "data/features_iep/features_meta.json"
INDEX_PATH = BASE_DIR / "data/index_iep/hedonic_index.parquet"

TMP_LOG_PATH = Path("logs/requests_tmp.jsonl")  # written per-request while app runs
PERSISTENT_LOG_PATH = Path("logs/requests.json")  # written on shutdown only
TMP_LOG_PATH.parent.mkdir(exist_ok=True)

app = FastAPI()
geo_cache = {}


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    messages = []
    for err in exc.errors():
        msg = err.get("msg", "")
        # Pydantic wraps our ValueError as "Value error, <our text>"
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, ") :]
        elif err.get("type") == "missing":
            loc = err.get("loc", [])
            field = loc[-1] if loc else "поле"
            label = _FIELD_LABELS.get(str(field), str(field))
            msg = f"Поле «{label}» обязательно"
        messages.append(msg)
    return JSONResponse(status_code=422, content={"detail": messages})


model = CatBoostRegressor()
model.load_model(MODEL_PATH)

continous_features, categorical_features = pl.read_json(FEATURES_NAMES_PATH)
continous_features = continous_features.item().to_list()
categorical_features = categorical_features.item().to_list()

# Hedonic price index: shown from 2025-06 onwards, rebased so 2025-06 = 100.
_index_df = pl.read_parquet(INDEX_PATH).filter(pl.col("year_month") >= "2025_06")
_index_base = _index_df["price_sq_forecast"][0]
_index_df = _index_df.with_columns(
    (100 * pl.col("price_sq_forecast") / _index_base).alias("index")
)
INDEX_PAYLOAD = {
    "periods": [p.replace("_", "-") for p in _index_df["year_month"].to_list()],
    "index": [round(v, 2) for v in _index_df["index"].to_list()],
    "price_sq_forecast": [
        round(v, 0) for v in _index_df["price_sq_forecast"].to_list()
    ],
}

# Per-flat hedonic path. The chosen flat's own features are held fixed and only
# the time parameter is varied: the pure time effect is taken from the global
# catboost hedonic index (the same index rebased so first shown month = 100).
# The flat's ₽/m² for each month is its current catboost estimate scaled by the
# index ratio to the latest month, so the latest point equals the present-day
# prediction shown to the user.
_INDEX_PERIODS = INDEX_PAYLOAD["periods"]
_INDEX_VALUES = INDEX_PAYLOAD["index"]  # rebased: first shown month = 100
_index_series = _index_df["index"].to_list()
_index_latest = _index_series[-1]
_INDEX_RATIO_TO_LATEST = [v / _index_latest for v in _index_series]


def _append_tmp_log(record: dict) -> None:
    with open(TMP_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _flush_to_persistent() -> None:
    """Read tmp JSONL and append all records to the persistent JSON array."""
    if not TMP_LOG_PATH.exists():
        return
    records = []
    with open(TMP_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not records:
        return
    existing = []
    if PERSISTENT_LOG_PATH.exists():
        with open(PERSISTENT_LOG_PATH, encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    existing.extend(records)
    with open(PERSISTENT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    TMP_LOG_PATH.unlink(missing_ok=True)
    logger.info(f"Flushed {len(records)} records → {PERSISTENT_LOG_PATH}")


_FIELD_LABELS = {
    "total_room_count": "Количество комнат",
    "total_square": "Общая площадь",
    "kitchen_square": "Площадь кухни",
    "life_square": "Жилая площадь",
    "balcony_count": "Количество балконов",
    "storey": "Этаж квартиры",
    "storeys_count": "Количество этажей в доме",
    "bathroom_count": "Количество санузлов",
    "elevator_count": "Количество лифтов",
    "built_year": "Год постройки",
    "lat": "Широта",
    "lon": "Долгота",
}


def _bad_chars(raw: str) -> str:
    """Return the non-numeric junk found in the string."""
    found = re.findall(r"[^\d.,\s\-]", raw)
    return "".join(dict.fromkeys(found)) or raw  # deduplicated, fallback to full string


class FlatRequest(BaseModel):
    total_room_count: int
    total_square: float
    kitchen_square: float
    life_square: float
    balcony_count: int
    is_apartment: int
    storey: int
    storeys_count: int
    bathroom_count: int
    elevator_count: int
    lat: float
    lon: float
    apartment_condition: str
    built_year: int
    building_batch_name: Optional[str] = None

    @field_validator(
        "total_square", "kitchen_square", "life_square", "lat", "lon", mode="before"
    )
    @classmethod
    def coerce_decimal(cls, v, info):
        raw = str(v).strip()
        label = _FIELD_LABELS.get(info.field_name, info.field_name)
        if not raw:
            raise ValueError(f"Заполните поле «{label}»")
        cleaned = raw.replace(",", ".").replace(" ", "")
        try:
            return float(cleaned)
        except ValueError:
            junk = _bad_chars(raw)
            raise ValueError(f"Уберите текст «{junk}» из поля «{label}»")

    @field_validator(
        "total_room_count",
        "balcony_count",
        "storey",
        "storeys_count",
        "bathroom_count",
        "elevator_count",
        "built_year",
        mode="before",
    )
    @classmethod
    def coerce_int(cls, v, info):
        raw = str(v).strip()
        label = _FIELD_LABELS.get(info.field_name, info.field_name)
        if not raw:
            raise ValueError(f"Заполните поле «{label}»")
        cleaned = raw.replace(",", ".").replace(" ", "")
        try:
            return int(float(cleaned))
        except ValueError:
            junk = _bad_chars(raw)
            raise ValueError(f"Уберите текст «{junk}» из поля «{label}»")


@app.post("/transform")
def transform(data: FlatRequest):
    start_time = datetime.utcnow()

    try:
        request_dict = data.model_dump()

        df = pl.DataFrame([request_dict])

        df = (
            df.with_columns(
                pl.lit(data.lat).alias("location.lat"),
                pl.lit(data.lon).alias("location.lon"),
                pl.lit("guid").alias("guid"),
            )
            .drop(["lat", "lon"])
            .with_columns(
                [
                    (pl.col("storey") == 1).cast(pl.Int8).alias("storey_first"),
                    (pl.col("storey") == pl.col("storeys_count"))
                    .cast(pl.Int8)
                    .alias("storey_last"),
                    (pl.col("storey") / pl.col("storeys_count")).alias(
                        "storey_relative"
                    ),
                ]
            )
            .with_columns(
                pl.lit("note").alias("note"),
                pl.lit("прямая продажа").alias("sale_type_name"),
            )
            .with_columns(pl.lit(20_000_000).alias("price"))
        )

        features = FeaturesService(df).calculate_features()
        features = features.select(continous_features + categorical_features)

        features = features.with_columns(
            [pl.col(categorical_features).fill_null("missing")]
        )

        pool = Pool(features, cat_features=categorical_features)

        pred = np.exp(model.predict(pool))
        price_sq = float(round(pred[0], 0))

        # Hedonic path for this exact flat: features fixed, only time varied.
        flat_price_sq = [round(price_sq * r, 0) for r in _INDEX_RATIO_TO_LATEST]

        response = {
            "price_per_m2": price_sq,
            "price_total": round(price_sq * data.total_square, 0),
            "hedonic": {
                "periods": _INDEX_PERIODS,
                "index": _INDEX_VALUES,
                "price_sq": flat_price_sq,
            },
        }

        latency = (datetime.utcnow() - start_time).total_seconds()

        record = {
            "timestamp": start_time.isoformat(),
            "lat": data.lat,
            "lon": data.lon,
            "input": request_dict,
            "prediction": response,
            "latency_sec": latency,
            "status": "ok",
        }

        _append_tmp_log(record)
        logger.info(f"SUCCESS | latency={latency:.3f}s | {response}")

        return response

    except Exception as e:
        traceback.print_exc()

        error_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "lat": getattr(data, "lat", None),
            "lon": getattr(data, "lon", None),
            "input": data.model_dump(),
            "error": str(e),
            "status": "error",
        }

        _append_tmp_log(error_record)
        logger.error(f"ERROR | {e}")

        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
def shutdown_event():
    _flush_to_persistent()


@app.get("/index")
def price_index():
    """Quality-adjusted hedonic price index (2025-06 = 100) and price/m² level."""
    return INDEX_PAYLOAD


@app.get("/geocode")
async def geocode(text: str = Query(..., min_length=2)):
    """Geocode an address string → lat/lon using the JS/Geocoder key."""
    params = {
        "apikey": base_settings.YANDEX_MAPS_API_JS_KEY,
        "geocode": text,
        "format": "json",
        "results": 1,
        "lang": "ru_RU",
    }
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.get("https://geocode-maps.yandex.ru/1.x/", params=params)
        r.raise_for_status()
        data = r.json()

    try:
        pos = data["response"]["GeoObjectCollection"]["featureMember"][0]["GeoObject"][
            "Point"
        ]["pos"]
        lon, lat = map(float, pos.split())
        return {"lat": lat, "lon": lon}
    except (KeyError, IndexError, ValueError):
        raise HTTPException(status_code=404, detail="Адрес не найден")


@app.get("/", response_class=HTMLResponse)
def form():
    building_options = '<option value="">— не знаю —</option>' + "".join(
        [f'<option value="{b}">{b}</option>' for b in config_api.BUILDING_BATCHES]
    )
    condition_options = "".join(
        [
            f'<option value="{c}" {"selected" if c == "среднее состояние" else ""}>{c}</option>'
            for c in config_api.APARTMENT_CONDITIONS
        ]
    )
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>TopFlip — Оценка реальной стоимости квартиры</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
<script src="https://api-maps.yandex.ru/2.1/?apikey={base_settings.YANDEX_MAPS_API_JS_KEY}&lang=ru_RU"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --blue:    #1a56db;
    --blue-dk: #1035a0;
    --green:   #059669;
    --red:     #dc2626;
    --bg:      #f1f5f9;
    --card:    #ffffff;
    --border:  #e2e8f0;
    --text:    #0f172a;
    --muted:   #64748b;
    --radius:  12px;
  }}

  body {{
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }}

  /* ── Hero ───────────────────────────────── */
  .hero {{
    position: relative;
    height: 320px;
    background:
      linear-gradient(to bottom, rgba(10,20,60,.55) 0%, rgba(10,20,60,.75) 100%),
      url('https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?auto=format&fit=crop&w=1920&q=80')
      center/cover no-repeat;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 0 24px;
    color: #fff;
  }}
  .hero-badge {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    background: rgba(255,255,255,.15);
    border: 1px solid rgba(255,255,255,.25);
    border-radius: 999px;
    padding: 4px 14px;
    margin-bottom: 16px;
    backdrop-filter: blur(4px);
  }}
  .hero h1 {{
    font-size: clamp(1.6rem, 4vw, 2.6rem);
    font-weight: 700;
    line-height: 1.2;
    max-width: 680px;
  }}
  .hero p {{
    margin-top: 10px;
    font-size: .95rem;
    opacity: .8;
    max-width: 520px;
  }}

  /* ── Main layout ────────────────────────── */
  .container {{
    max-width: 1280px;
    margin: 0 auto;
    padding: 36px 24px 60px;
  }}
  .main-grid {{
    display: grid;
    grid-template-columns: 1fr 400px;
    gap: 28px;
    align-items: start;
  }}
  @media (max-width: 900px) {{
    .main-grid {{ grid-template-columns: 1fr; }}
  }}

  /* ── Card ───────────────────────────────── */
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
  }}
  .section-title {{
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }}

  /* ── Form grid ──────────────────────────── */
  .form-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0 24px;
  }}
  .form-group {{
    margin-bottom: 16px;
  }}
  .form-group.full {{ grid-column: 1 / -1; }}
  label {{
    display: block;
    font-size: .78rem;
    font-weight: 500;
    color: var(--muted);
    margin-bottom: 5px;
  }}
  input, select {{
    width: 100%;
    padding: 9px 12px;
    border: 1.5px solid var(--border);
    border-radius: 8px;
    font-family: inherit;
    font-size: .9rem;
    color: var(--text);
    background: #fff;
    transition: border-color .15s;
    outline: none;
  }}
  input:focus, select:focus {{ border-color: var(--blue); }}
  input[readonly] {{ background: #f8fafc; cursor: default; color: var(--muted); }}

  .divider {{ margin: 24px 0 20px; }}

  /* ── Map section ───────────────────────── */
  .map-section {{
    margin-top: 28px;
  }}
  .map-coords-row {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
  }}
  .map-coords-row label {{
    white-space: nowrap;
    margin: 0;
  }}
  .map-coords-row input {{
    max-width: 300px;
  }}
  #map {{
    width: 100%;
    height: 560px;
    border-radius: 10px;
    overflow: hidden;
    border: 1.5px solid var(--border);
  }}

  /* ── Sidebar ────────────────────────────── */
  .sidebar {{ display: flex; flex-direction: column; gap: 20px; }}
 
  .btn {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    width: 100%;
    padding: 14px;
    background: var(--blue);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-family: inherit;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: background .15s, transform .1s;
  }}
  .btn:hover {{ background: var(--blue-dk); }}
  .btn:active {{ transform: scale(.98); }}
  .btn:disabled {{ background: #94a3b8; cursor: not-allowed; }}

  /* ── Result card ────────────────────────── */
  #result-card {{
    display: none;
    border-radius: var(--radius);
    overflow: hidden;
    animation: fadeUp .3s ease;
  }}
  @keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}
  .result-header {{
    background: var(--green);
    color: #fff;
    padding: 14px 20px;
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
  }}
  .result-body {{
    background: #fff;
    border: 1px solid var(--border);
    border-top: none;
    border-radius: 0 0 var(--radius) var(--radius);
    padding: 20px;
  }}
  .result-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 8px 0;
  }}
  .result-row:not(:last-child) {{ border-bottom: 1px solid var(--border); }}
  .result-label {{ font-size: .85rem; color: var(--muted); }}
  .result-value {{ font-size: 1.25rem; font-weight: 700; color: var(--text); }}
  .result-value.big {{ font-size: 1.6rem; color: var(--green); }}

  /* ── Error box ──────────────────────────── */
  #error-box {{
    display: none;
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: var(--radius);
    padding: 16px 20px;
    animation: fadeUp .25s ease;
  }}
  #error-box .err-title {{
    font-size: .8rem;
    font-weight: 600;
    color: var(--red);
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: .06em;
  }}
  #error-list {{ list-style: none; }}
  #error-list li {{
    font-size: .88rem;
    color: #991b1b;
    padding: 3px 0;
    padding-left: 14px;
    position: relative;
  }}
  #error-list li::before {{
    content: '•';
    position: absolute;
    left: 0;
    color: var(--red);
  }}

  /* ── Info tip ───────────────────────────── */
  .tip {{
    font-size: .78rem;
    color: var(--muted);
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    line-height: 1.5;
  }}

  /* ── Charts ─────────────────────────────── */
  .charts-section {{ margin-top: 28px; }}
  .charts-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 28px;
  }}
  @media (max-width: 900px) {{
    .charts-grid {{ grid-template-columns: 1fr; }}
  }}
  .chart-box {{
    display: flex;
    flex-direction: column;
  }}
  .chart-box h3 {{
    font-size: .9rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 4px;
  }}
  .chart-box .chart-sub {{
    font-size: .76rem;
    color: var(--muted);
    margin-bottom: 12px;
  }}
  .chart-canvas-wrap {{
    position: relative;
    height: 320px;
  }}
</style>
</head>
<body>

<div class="hero">
  <h1>Оценка стоимости вторичной квартиры в Москве и области</h1>
  <p>Укажите параметры квартиры и кликните на карте — модель рассчитает рыночную цену</p>
</div>

<div class="container">
<form id="form">
<div class="main-grid">

  <!-- ── LEFT: form ── -->
  <div>
    <div class="card">
      <div class="section-title">Планировка</div>
      <div class="form-grid">
        <div class="form-group">
          <label>Комнат</label>
          <input name="total_room_count" value="3" placeholder="3"/>
        </div>
        <div class="form-group">
          <label>Общая площадь, м²</label>
          <input name="total_square" value="60" placeholder="60"/>
        </div>
        <div class="form-group">
          <label>Жилая площадь, м²</label>
          <input name="life_square" value="42" placeholder="42"/>
        </div>
        <div class="form-group">
          <label>Площадь кухни, м²</label>
          <input name="kitchen_square" value="8" placeholder="8"/>
        </div>
        <div class="form-group">
          <label>Балконов / лоджий</label>
          <input name="balcony_count" value="2" placeholder="0"/>
        </div>
        <div class="form-group">
          <label>Санузлов</label>
          <input name="bathroom_count" value="1" placeholder="1"/>
        </div>
      </div>

      <div class="divider section-title">Дом</div>
      <div class="form-grid">
        <div class="form-group">
          <label>Этаж квартиры</label>
          <input name="storey" value="2" placeholder="2"/>
        </div>
        <div class="form-group">
          <label>Этажей в доме</label>
          <input name="storeys_count" value="12" placeholder="12"/>
        </div>
        <div class="form-group">
          <label>Лифтов</label>
          <input name="elevator_count" value="2" placeholder="2"/>
        </div>
        <div class="form-group">
          <label>Год постройки</label>
          <input name="built_year" value="1980" placeholder="1980"/>
        </div>
        <div class="form-group">
          <label>Серия дома</label>
          <select name="building_batch_name">{building_options}</select>
        </div>
        <div class="form-group">
          <label>Состояние квартиры</label>
          <select name="apartment_condition">{condition_options}</select>
        </div>
      </div>

    </div>
  </div>

  <!-- ── RIGHT: sidebar ── -->
  <div class="sidebar">
    <div class="card">
      <div class="section-title">Расчёт цены</div>
      <div class="tip">
        Модель оценивает стоимость на основе параметров квартиры и её расположения.
        Точность выше в пределах МКАД и ближнего Подмосковья.
      </div>
      <br/>
      <button class="btn" type="submit" id="submit-btn">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
          <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
        Рассчитать стоимость
      </button>
    </div>

    <div id="error-box">
      <div class="err-title">Проверьте поля</div>
      <ul id="error-list"></ul>
    </div>

    <div id="result-card">
      <div class="result-header">Результат оценки</div>
      <div class="result-body">
        <div class="result-row">
          <span class="result-label">Цена за м²</span>
          <span class="result-value" id="r-sqm"></span>
        </div>
        <div class="result-row">
          <span class="result-label">Итоговая цена</span>
          <span class="result-value big" id="r-total"></span>
        </div>
      </div>
    </div>
  </div>

</div>

  <!-- ── MAP: full-width below grid ── -->
  <div class="map-section card">
    <div class="section-title">Местоположение — найдите адрес или кликните на карте</div>

    <div class="map-coords-row">
      <label>Координаты:</label>
      <input id="coords" readonly placeholder="Кликните на карту или выберите адрес"/>
    </div>
    <input type="hidden" id="lat" name="lat" value="55.751244"/>
    <input type="hidden" id="lon" name="lon" value="37.618423"/>
    <div id="map"></div>
  </div>

</form>

  <!-- ── CHARTS: per-flat hedonic index ── -->
  <div class="charts-section card">
    <div class="section-title">Гедонический индекс вашей квартиры</div>
    <div id="charts-placeholder" class="tip">
      Рассчитайте стоимость своей квартиры — модель зафиксирует её характеристики
      и покажет, как менялась бы её цена во времени при изменении только временного
      параметра.
    </div>
    <div class="charts-grid" id="charts-grid" style="display:none;">
      <div class="chart-box">
        <h3>Индекс стоимости вашей квартиры</h3>
        <div class="chart-sub">База: июнь 2025 = 100</div>
        <div class="chart-canvas-wrap"><canvas id="flatIndexChart"></canvas></div>
      </div>
      <div class="chart-box">
        <h3>Цена за м² вашей квартиры, ₽</h3>
        <div class="chart-sub">По месяцам, характеристики зафиксированы</div>
        <div class="chart-canvas-wrap"><canvas id="flatPriceChart"></canvas></div>
      </div>
    </div>
    <div class="tip" style="margin-top:18px;">
      Индекс изолирует чистый эффект времени: характеристики именно вашей квартиры
      фиксируются, и модель CatBoost оценивает, сколько она стоила бы за м² в каждом
      месяце. Последняя точка совпадает с текущей оценкой выше.
    </div>
  </div>

</div>

<script>
ymaps.ready(function () {{
  const defaultCoords = [55.751244, 37.618423];
  const map = new ymaps.Map("map", {{
    center: defaultCoords,
    zoom: 10,
    controls: ['zoomControl', 'geolocationControl']
  }});

  let placemark = new ymaps.Placemark(defaultCoords, {{}}, {{
    preset: 'islands#blueDotIcon'
  }});
  map.geoObjects.add(placemark);
  document.getElementById("coords").value =
    defaultCoords[0].toFixed(6) + ", " + defaultCoords[1].toFixed(6);

  function setPoint(coords) {{
    placemark.geometry.setCoordinates(coords);
    document.getElementById("coords").value =
      coords[0].toFixed(6) + ", " + coords[1].toFixed(6);
    document.getElementById("lat").value = coords[0];
    document.getElementById("lon").value = coords[1];
  }}

  map.events.add('click', e => setPoint(e.get('coords')));

  // Search control on the map
  const search = new ymaps.control.SearchControl({{
    options: {{
      float: 'left',
      noPlacemark: true,
      boundedBy: [[54.0, 35.0], [57.5, 40.5]],
      strictBounds: false,
      results: 5
    }}
  }});
  map.controls.add(search);
  search.events.add('resultselect', function (e) {{
    const idx = e.get('index');
    search.getResult(idx).then(function (res) {{
      const coords = res.geometry.getCoordinates();
      setPoint(coords);
      map.setCenter(coords, 15, {{ duration: 400 }});
    }});
  }});

  // Geolocation button
  map.controls.get('geolocationControl').events.add('locationchange', function (e) {{
    setPoint(e.get('position'));
    map.setCenter(e.get('position'), 15);
  }});

}});

function fmt(x, step) {{
  return (Math.round(x / step) * step).toLocaleString('ru-RU') + ' ₽';
}}

// ── Per-flat hedonic charts (styled to match the page) ──
const CHART_BLUE  = '#1a56db';
const CHART_GREEN = '#059669';
const CHART_GRID  = '#e2e8f0';
const CHART_MUTED = '#64748b';
let flatIndexChart = null;
let flatPriceChart = null;

Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.color = CHART_MUTED;

function baseAxes(yTitle, tickFmt) {{
  return {{
    x: {{ grid: {{ color: CHART_GRID }}, ticks: {{ maxRotation: 45, minRotation: 45 }} }},
    y: {{
      grid: {{ color: CHART_GRID }},
      title: {{ display: true, text: yTitle, color: CHART_MUTED }},
      ticks: {{ callback: tickFmt }}
    }}
  }};
}}

// Build/refresh the two per-flat charts from the /transform hedonic payload.
function renderFlatCharts(h) {{
  document.getElementById('charts-placeholder').style.display = 'none';
  document.getElementById('charts-grid').style.display = 'grid';

  // Chart 1 — flat's own hedonic index (June 2025 = 100)
  const indexCfg = {{
    type: 'line',
    data: {{
      labels: h.periods,
      datasets: [{{
        label: 'Индекс', data: h.index,
        borderColor: CHART_BLUE, backgroundColor: 'rgba(26,86,219,.08)',
        borderWidth: 2.5, pointRadius: 3, pointBackgroundColor: CHART_BLUE,
        tension: .25, fill: true
      }}]
    }},
    options: {{
      maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: baseAxes('Индекс (база = 100)', v => v)
    }}
  }};

  // Chart 2 — flat's own price per m² over time
  const priceCfg = {{
    type: 'line',
    data: {{
      labels: h.periods,
      datasets: [{{
        label: 'Цена за м² вашей квартиры', data: h.price_sq,
        borderColor: CHART_GREEN, backgroundColor: 'rgba(5,150,105,.08)',
        borderWidth: 2.5, pointRadius: 3, pointBackgroundColor: CHART_GREEN,
        tension: .25, fill: true
      }}]
    }},
    options: {{
      maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: baseAxes('₽ / м²', v => (v / 1000).toFixed(0) + 'k')
    }}
  }};

  if (flatIndexChart) {{
    flatIndexChart.data = indexCfg.data;
    flatIndexChart.update();
  }} else {{
    flatIndexChart = new Chart(document.getElementById('flatIndexChart'), indexCfg);
  }}
  if (flatPriceChart) {{
    flatPriceChart.data = priceCfg.data;
    flatPriceChart.update();
  }} else {{
    flatPriceChart = new Chart(document.getElementById('flatPriceChart'), priceCfg);
  }}
}}

document.getElementById("form").onsubmit = async (e) => {{
  e.preventDefault();
  const btn = document.getElementById("submit-btn");
  const errorBox = document.getElementById("error-box");
  const errorList = document.getElementById("error-list");
  const resultCard = document.getElementById("result-card");

  btn.disabled = true;
  btn.textContent = "Считаем…";
  errorBox.style.display = "none";
  resultCard.style.display = "none";

  const data = Object.fromEntries(new FormData(e.target));
  data.is_apartment = 0;
  if (data.building_batch_name === "") data.building_batch_name = null;

  try {{
    const res = await fetch("/transform", {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify(data)
    }});
    const r = await res.json();

    if (!res.ok) {{
      const msgs = Array.isArray(r.detail) ? r.detail : [r.detail || "Неизвестная ошибка"];
      errorList.innerHTML = msgs.map(m => '<li>' + m + '</li>').join("");
      errorBox.style.display = "block";
    }} else {{
      document.getElementById("r-sqm").textContent = fmt(r.price_per_m2, 1000);
      document.getElementById("r-total").textContent = fmt(r.price_total, 100000);
      resultCard.style.display = "block";
      if (r.hedonic) renderFlatCharts(r.hedonic);
    }}
  }} catch (err) {{
    errorList.innerHTML = '<li>Ошибка соединения: ' + err.message + '</li>';
    errorBox.style.display = "block";
  }} finally {{
    btn.disabled = false;
    btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Рассчитать стоимость`;
  }}
}};
</script>
</body>
</html>"""
