"""
EvapTool 2.0 - Streamlit web version
Behnam Zamani

Downloads recent daily climate data from the closest DWD weather station
(DWD Open Data), fills missing/insufficient columns from nearby stations,
adds wind data from the closest hourly wind station if needed, and
calculates open-water evaporation (Penman, Priestley-Taylor) and
reference evapotranspiration ET0 (FAO Penman-Monteith).
"""

import io
import math
import re
import zipfile

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =====================================================
# SETTINGS
# =====================================================

KL_RECENT_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "observations_germany/climate/daily/kl/recent/"
)
KL_STATION_FILE = "KL_Tageswerte_Beschreibung_Stationen.txt"

WIND_RECENT_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "observations_germany/climate/hourly/wind/recent/"
)
WIND_STATION_FILE = "FF_Stundenwerte_Beschreibung_Stationen.txt"

MISSING_VALUES = [-999, -999.0, -9999, -9999.0]
REQUIRED_COLUMNS = ["RSK", "VPM", "PM", "TMK", "UPM", "TXK", "TNK"]

# How many nearby stations to try when replacing a bad column
MAX_REPLACEMENT_CANDIDATES = 10

CACHE_TTL = 6 * 3600  # downloaded DWD files are reused for 6 hours

# --- Model constants (unchanged from the original script) ---
GAMMA = 0.067      # kPa/degC psychrometric constant, elevation 0-100 m (FAO Annex 2, table 2.2)
Z0 = 0.03          # cm, roughness of water surface (Chow, Table 2.8.2)
Z2 = 2             # m, height of wind speed measurement
ALPHA_EVAP = 0.8   # Priestley-Taylor coefficient (Chow eq. 3.5.27)
RHO_W = 997        # kg/m3 density of water
K_VK = 0.4         # von Karman constant (Chow eq. 3.5.14)
KRS = 0.16         # 0.16 inland, 0.19 coastal
G = 0              # soil heat flux, ignored for daily calculations
LAMBDA = 2.45      # MJ/kg latent heat of vaporization


# =====================================================
# DOWNLOAD HELPERS (cached)
# =====================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def fetch_text(url: str, encoding: str = "latin1") -> str:
    return fetch_bytes(url).decode(encoding)


def clean_dwd(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    df["MESS_DATUM"] = df["MESS_DATUM"].astype(str).str.strip()
    return df.replace(MISSING_VALUES, np.nan)


def read_product_from_zip(zip_bytes: bytes, prefix: str) -> pd.DataFrame:
    """Read the 'produkt_...' text file from a DWD zip (in memory, no disk)."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = [n for n in z.namelist() if n.startswith(prefix) and n.endswith(".txt")]
        if not names:
            raise FileNotFoundError(f"No {prefix}* file found in zip.")
        with z.open(names[0]) as f:
            df = pd.read_csv(f, sep=";", low_memory=False, encoding="latin1")
    return clean_dwd(df)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_station_list(url: str, name_end: int) -> pd.DataFrame:
    """Parse a DWD station description file."""
    stations = []
    for line in fetch_text(url).splitlines():
        if line.startswith("Stations_id") or line.startswith("-") or not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        try:
            stations.append({
                "Stations_id": parts[0].zfill(5),
                "von_datum": parts[1],
                "bis_datum": parts[2],
                "Stationshoehe": float(parts[3]),
                "geoBreite": float(parts[4]),
                "geoLaenge": float(parts[5]),
                "Stationsname": " ".join(parts[6:name_end]),
            })
        except Exception:
            continue
    df = pd.DataFrame(stations)
    df["bis_datum"] = pd.to_datetime(df["bis_datum"], format="%Y%m%d", errors="coerce")
    return df.dropna(subset=["bis_datum"])


def load_kl_recent(code: str) -> pd.DataFrame:
    zip_bytes = fetch_bytes(KL_RECENT_URL + f"tageswerte_KL_{code}_akt.zip")
    return read_product_from_zip(zip_bytes, "produkt_klima_tag_")


def column_is_good(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    values = pd.to_numeric(df[col], errors="coerce")
    valid, nan = values.notna().sum(), values.isna().sum()
    return valid > 0 and valid > nan


def add_distance(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    df = df.copy()
    df["distance"] = np.sqrt((df["geoBreite"] - lat) ** 2 + (df["geoLaenge"] - lon) ** 2)
    return df.sort_values("distance").reset_index(drop=True)


# =====================================================
# DATA PIPELINE
# =====================================================

def get_dwd_data(lat_in: float, lon_in: float, log) -> dict:
    # ---- Station list -----------------------------------------------
    log("Loading recent DWD station information...")
    stations_df = add_distance(
        load_station_list(KL_RECENT_URL + KL_STATION_FILE, -2), lat_in, lon_in
    )

    # ---- Closest station with recent data (1 month, else 2 months) --
    closest = None
    used_months = None
    for months_back in [1, 2]:
        cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(months=months_back)
        cand = stations_df[stations_df["bis_datum"] >= cutoff]
        if len(cand):
            closest = cand.iloc[0]
            used_months = months_back
            break
    if closest is None:
        raise RuntimeError("No DWD climate station found with data within the last 1 or 2 months.")

    station_code = closest["Stations_id"]
    log(f"Closest station: **{closest['Stationsname']}** ({station_code}), "
        f"last data {closest['bis_datum'].date()}")

    # ---- Recent data of the closest station -------------------------
    log("Downloading recent climate data...")
    df_dwd = load_kl_recent(station_code)

    # ---- Replace bad / missing columns from nearby stations ---------
    bad_columns = [c for c in REQUIRED_COLUMNS if not column_is_good(df_dwd, c)]
    replacement_info = {}

    if bad_columns:
        log(f"Columns to replace from nearby stations: {bad_columns}")
        cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(months=2)
        candidates = stations_df[
            (stations_df["Stations_id"] != station_code)
            & (stations_df["bis_datum"] >= cutoff)
        ].head(MAX_REPLACEMENT_CANDIDATES)

        for missing_col in bad_columns:
            found = False
            for _, cs in candidates.iterrows():
                code, name = cs["Stations_id"], cs["Stationsname"]
                try:
                    log(f"Checking {name} ({code}) for {missing_col}...")
                    df_c = load_kl_recent(code)
                    if missing_col not in df_c.columns:
                        continue
                    df_c[missing_col] = pd.to_numeric(df_c[missing_col], errors="coerce")
                    temp_col = f"{missing_col}_replacement"
                    df_rep = (df_c[["MESS_DATUM", missing_col]]
                              .drop_duplicates(subset="MESS_DATUM", keep="last")
                              .rename(columns={missing_col: temp_col}))
                    test = df_dwd[["MESS_DATUM"]].merge(df_rep, on="MESS_DATUM", how="left")
                    valid = int(test[temp_col].notna().sum())
                    nan = int(test[temp_col].isna().sum())
                    if valid > 0 and valid > nan:
                        df_dwd = (df_dwd.drop(columns=[missing_col], errors="ignore")
                                  .merge(df_rep, on="MESS_DATUM", how="left")
                                  .rename(columns={temp_col: missing_col}))
                        replacement_info[missing_col] = {
                            "Station": name, "Code": code,
                            "Valid days": valid, "Missing days": nan,
                        }
                        log(f"{missing_col} replaced from {name} ({code})")
                        found = True
                        break
                except Exception as e:
                    log(f"Station {code} failed: {e}")
            if not found:
                log(f"⚠️ No suitable replacement found for {missing_col}.")

    # ---- Wind (FM) from closest hourly wind station if needed -------
    wind_info = None
    if "FM" not in df_dwd.columns or df_dwd["FM"].notna().sum() < df_dwd["FM"].isna().sum():
        df_dwd = df_dwd.drop(columns=["FM", "FX", "QN_3", "QN_4"], errors="ignore")
        log("Daily wind (FM) insufficient - searching closest hourly wind station...")

        wind_df = load_station_list(WIND_RECENT_URL + WIND_STATION_FILE, -1)
        recent_codes = set(re.findall(r"stundenwerte_FF_(\d{5})_akt\.zip",
                                      fetch_text(WIND_RECENT_URL)))
        wind_df = wind_df[wind_df["Stations_id"].isin(recent_codes)]
        if wind_df.empty:
            raise RuntimeError("No wind stations with recent data were found.")

        # Distance from the selected climate station (as in the original script)
        wind_df = add_distance(wind_df, float(closest["geoBreite"]), float(closest["geoLaenge"]))
        ws = wind_df.iloc[0]
        log(f"Closest wind station: {ws['Stationsname']} ({ws['Stations_id']})")

        df_wind = read_product_from_zip(
            fetch_bytes(WIND_RECENT_URL + f"stundenwerte_FF_{ws['Stations_id']}_akt.zip"),
            "produkt_ff_stunde_",
        )
        df_wind["MESS_DATUM"] = pd.to_datetime(df_wind["MESS_DATUM"], format="%Y%m%d%H",
                                               errors="coerce")
        speed_col = "F" if "F" in df_wind.columns else ("FF" if "FF" in df_wind.columns else None)
        if speed_col is None:
            raise KeyError("No wind-speed column ('F' or 'FF') found.")
        df_wind[speed_col] = pd.to_numeric(df_wind[speed_col], errors="coerce")

        df_wind_daily = (df_wind.dropna(subset=["MESS_DATUM"])
                         .set_index("MESS_DATUM").resample("D")
                         .agg({speed_col: "mean"}).reset_index()
                         .rename(columns={speed_col: "FM"}))
        df_wind_daily["MESS_DATUM"] = df_wind_daily["MESS_DATUM"].dt.strftime("%Y%m%d")
        df_dwd = df_dwd.merge(df_wind_daily[["MESS_DATUM", "FM"]], on="MESS_DATUM", how="left")
        wind_info = {"Station": ws["Stations_id"], "Name": ws["Stationsname"]}
        log("Hourly wind data merged.")

    return {
        "df": df_dwd,
        "station": closest,
        "used_months": used_months,
        "replacements": replacement_info,
        "wind": wind_info,
    }


# =====================================================
# EVAPORATION CALCULATION
# =====================================================

def compute_evaporation(df: pd.DataFrame, lat: float, elev_st: float) -> pd.DataFrame:
    df = df.copy()
    for c in REQUIRED_COLUMNS + ["FM"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    pi = math.pi
    df["date"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d")

    # Vapour pressure (FAO chapter 3, box 7)
    df["eTmax"] = 0.6108 * np.exp((17.27 * df["TXK"]) / (df["TXK"] + 237.3))  # kPa
    df["eTmin"] = 0.6108 * np.exp((17.27 * df["TNK"]) / (df["TNK"] + 237.3))  # kPa
    df["es"] = 0.1 * df["VPM"] / (df["UPM"] / 100)            # saturation vapour pressure kPa
    df["vap_deficit"] = df["es"] - df["VPM"] * 0.1             # es - ea, kPa

    # Slope of saturation vapour pressure curve (FAO eq. 13)
    # FIX: original had exp((17.27*T)/T+273.3) -> parenthesis error and 273.3 instead of 237.3
    df["delta"] = (4098 * (0.6108 * np.exp((17.27 * df["TMK"]) / (df["TMK"] + 237.3)))) \
        / ((df["TMK"] + 237.3) ** 2)

    # Solar radiation (FAO chapter 3)
    df["doy"] = df["date"].dt.dayofyear
    df["dr"] = 1 + 0.033 * np.cos(2 * pi * df["doy"] / 365)
    df["dlt"] = 0.409 * np.sin(2 * pi * df["doy"] / 365 - 1.39)
    phi = lat * pi / 180
    df["omega_s"] = np.arccos(-np.tan(phi) * np.tan(df["dlt"]))
    df["Ra"] = (24 * 60) * 0.0820 * df["dr"] * (
        df["omega_s"] * np.sin(phi) * np.sin(df["dlt"])
        + np.cos(phi) * np.cos(df["dlt"]) * np.sin(df["omega_s"])) / pi   # MJ m-2 d-1
    df["N"] = (24 / pi) * df["omega_s"]
    df["Rso"] = (0.75 + 2e-5 * elev_st) * df["Ra"]
    temp_range = (df["TXK"] - df["TNK"]).clip(lower=0)
    df["Rs"] = np.minimum(KRS * np.sqrt(temp_range) * df["Ra"], df["Rso"])
    df["Rns"] = (1 - 0.23) * df["Rs"]

    # Net longwave radiation (FAO eq. 39)
    # FIX: temperatures must be raised to the 4th power (Stefan-Boltzmann)
    df["Rnl"] = 4.903e-9 * (((df["TXK"] + 273.16) ** 4 + (df["TNK"] + 273.16) ** 4) / 2) \
        * (0.34 - 0.14 * np.sqrt(df["VPM"] * 0.1)) * ((1.35 * df["Rs"] / df["Rso"]) - 0.35)
    df["Rn"] = df["Rns"] - df["Rnl"]                            # MJ m-2 d-1

    # Reference evapotranspiration ET0 (FAO Penman-Monteith, mm/d)
    df["ET0"] = (0.408 * df["delta"] * (df["Rn"] - G)
                 + GAMMA * (900 / (df["TMK"] + 273)) * df["FM"] * df["vap_deficit"]) \
        / (df["delta"] + GAMMA * (1 + 0.34 * df["FM"]))

    # Open-water evaporation, Penman (Chow eqs. 3.5.10-3.5.26)
    df["rho_a"] = (df["PM"] * 100 * 0.0289652 + df["VPM"] * 100 * 0.018016) \
        / (8.31446 * (df["TMK"] + 273))
    df["B"] = (0.622 * (K_VK ** 2) * df["rho_a"] * df["FM"]) \
        / (df["PM"] * 100 * RHO_W * (np.log(Z2 * 100 / Z0)) ** 2)
    # FIX: Chow eq. 3.5.17 gives m/s -> convert to mm/d
    df["Ea"] = df["B"] * (df["es"] * 1000 - df["VPM"] * 100) * 86400 * 1000
    # FIX: Rn is in MJ m-2 d-1, so Er [mm/d] = Rn / lambda = 0.408*Rn
    #      (0.0353 is the factor for Rn in W/m2)
    df["Er"] = df["Rn"] / LAMBDA

    df["Eo_penman"] = (df["delta"] / (df["delta"] + GAMMA)) * df["Er"] \
        + (GAMMA / (df["delta"] + GAMMA)) * df["Ea"]
    df["Eo_penman"] = df["Eo_penman"].mask(df["Eo_penman"].lt(0), 0)

    # Priestley-Taylor (Chow eq. 3.5.27)
    df["Eo_prst"] = ALPHA_EVAP * (df["delta"] / (df["delta"] + GAMMA)) * df["Er"]
    df["Eo_prst"] = df["Eo_prst"].mask(df["Eo_prst"].lt(0), 0)

    df["water_bal_penman"] = df["RSK"] - df["Eo_penman"]
    df["water_bal_prst"] = df["RSK"] - df["Eo_prst"]
    return df


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


# =====================================================
# USER INTERFACE
# =====================================================

st.set_page_config(page_title="EvapTool", page_icon="💧", layout="wide")

st.title("💧 EvapTool")
st.markdown(
    "Downloads recent meteorological data from the closest "
    "[DWD](https://opendata.dwd.de) weather stations to your location and "
    "calculates open-water evaporation and reference evapotranspiration.  \n"
    "*by Behnam Zamani*"
)

with st.form("input"):
    coord_str = st.text_input(
        "Coordinates as Latitude, Longitude (WGS84)",
        value="52.3759, 9.7320",
        help="Example: 52.3759, 9.7320 (Hannover). Locations must be in Germany.",
    )
    submitted = st.form_submit_button("Calculate evaporation", type="primary")

if submitted:
    try:
        lat_in, lon_in = [float(x.strip()) for x in coord_str.split(",")]
    except Exception:
        st.error("Invalid coordinate format. Example: 52.3759, 9.7320")
        st.stop()

    if not (47.0 <= lat_in <= 55.5 and 5.5 <= lon_in <= 15.5):
        st.warning("These coordinates seem to be outside Germany. "
                   "DWD stations only cover Germany, so results may not be meaningful.")

    with st.status("Collecting DWD data...", expanded=True) as status:
        try:
            data = get_dwd_data(lat_in, lon_in, st.write)
            stn = data["station"]
            st.write("Calculating evaporation...")
            df_res = compute_evaporation(data["df"], float(stn["geoBreite"]),
                                         float(stn["Stationshoehe"]))
            status.update(label="Calculation completed", state="complete", expanded=False)
            st.session_state["result"] = {**data, "df": df_res,
                                          "input": (lat_in, lon_in)}
        except Exception as e:
            status.update(label="Calculation failed", state="error")
            st.error(f"Error: {e}")
            st.stop()

res = st.session_state.get("result")
if res:
    df = res["df"]
    stn = res["station"]
    code = stn["Stations_id"]

    st.subheader("Weather station")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Station", stn["Stationsname"])
    c2.metric("Station code", code)
    c3.metric("Last data", str(stn["bis_datum"].date()))
    c4.metric("Elevation", f"{stn['Stationshoehe']:.0f} m")

    st.map(pd.DataFrame({
        "lat": [res["input"][0], stn["geoBreite"]],
        "lon": [res["input"][1], stn["geoLaenge"]],
        "color": ["#1f77b4", "#d62728"],
    }), color="color", size=300, zoom=8)
    st.caption("🔵 your location  🔴 DWD station")

    if res["replacements"]:
        st.markdown("**Columns replaced from nearby stations:**")
        st.dataframe(pd.DataFrame(res["replacements"]).T, use_container_width=True)
    if res["wind"]:
        st.markdown(f"**Wind data from hourly station:** {res['wind']['Name']} "
                    f"({res['wind']['Station']})")

    st.subheader("Results")
    period = f"{df['date'].min().date()} – {df['date'].max().date()}"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Period", period)
    m2.metric("Σ Eo Penman", f"{df['Eo_penman'].sum():.0f} mm")
    m3.metric("Σ Eo Priestley-Taylor", f"{df['Eo_prst'].sum():.0f} mm")
    m4.metric("Σ Precipitation", f"{df['RSK'].sum():.0f} mm")

    tab1, tab2, tab3 = st.tabs(["Daily", "Monthly", "Data table"])
    plot_cols = {"Eo_penman": "Eo Penman", "Eo_prst": "Eo Priestley-Taylor", "ET0": "ET0"}
    with tab1:
        st.line_chart(df.set_index("date")[list(plot_cols)].rename(columns=plot_cols),
                      y_label="mm/day")
    with tab2:
        monthly = (df.set_index("date")[["RSK", "Eo_penman", "Eo_prst", "ET0"]]
                   .resample("MS").sum(min_count=1))
        monthly.index = monthly.index.strftime("%Y-%m")
        st.bar_chart(monthly[["Eo_penman", "Eo_prst", "RSK"]]
                     .rename(columns={"RSK": "Precipitation", **plot_cols}),
                     stack=False, y_label="mm/month")
        st.dataframe(monthly.round(1), use_container_width=True)
    with tab3:
        st.dataframe(df, use_container_width=True, height=400)

    st.download_button(
        "⬇️ Download results (Excel)",
        data=to_excel_bytes(df.assign(date=df["date"].dt.date)),
        file_name=f"res_Evap_DWD_station_{code}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )