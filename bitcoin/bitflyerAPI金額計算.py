#!/usr/bin/env python
# -*- coding: utf-8 -*-
# app.py — bitFlyer 取引照合 + Dash ダッシュボード（FINAL v4 with JPY-match filter）
# 2025‑04‑18

import os, io, textwrap, logging, hmac, hashlib, time
from urllib.parse import urlencode
import requests
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Dash, html, dash_table
from tabulate import tabulate

# ---------- 0. API & ログ ----------
API_URL    = "https://api.bitflyer.com"
API_KEY    = os.getenv("BITFLYER_API_KEY", "")
API_SECRET = os.getenv("BITFLYER_API_SECRET", "")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

# ---------- 1. 日本語 ↔ 英語 対応表 ----------
JP2EN = {"買い":"BUY", "売り":"SELL", "入金":"DEPOSIT",
         "出金":"WITHDRAW", "出金手数料":"WITHDRAW_FEE"}
EN2JP = {v:k for k,v in JP2EN.items()}

# ---------- 2. 手入力 RAW (CSV) ----------
RAW = textwrap.dedent(r"""取引日時,取引種別,価格,通貨1,数量1,手数料,通貨2,数量2,区分,注文ID
2025/04/17 11:15:30,買い,12024333,BTC,0.00308217,-0.00000461,JPY,-37062,媒介,JOR20250417-021352-711013
2025/04/17 11:15:30,買い,12024333,BTC,0.00107606,-0.00000162,JPY,-12938,媒介,JOR20250417-021352-711013
2025/04/17 11:01:52,売り,12048858,BTC,-0.00497973,-0.00000746,JPY,60000,媒介,JOR20250417-020149-365973
2025/04/17 10:03:59,買い,12382597,BTC,0.00403792,-,JPY,-50000,自己,BFF20250417-010359-363673
2025/04/17 10:02:45,入金,-,JPY,50000,-,-,-,-,MDP20250417-010245-327035
2025/04/17 09:49:13,買い,311.11,XRP,9.122175,-,JPY,-2838,自己,BFF20250417-004913-051875
2025/02/03 12:43:40,買い,14681999,BTC,0.014,-0.0000196,JPY,-205548,媒介,JOR20250203-034340-682920
2025/02/03 12:41:27,売り,14731800,BTC,-0.007,-0.0000098,JPY,103122,媒介,JOR20250203-034127-610825
2025/02/03 12:06:34,入金,-,JPY,50000,-,-,-,-,MDP20250203-030634-697089
2025/01/28 04:06:12,入金,-,JPY,50000,-,-,-,-,MDP20250127-190612-534708
2025/01/27 16:07:20,買い,15939621,BTC,0.00941051,-,JPY,-150000,自己,BFF20250127-070720-032249
2025/01/14 08:07:06,入金,-,JPY,50000,-,-,-,-,MDP20250113-230706-146535
2025/01/10 01:36:28,入金,-,JPY,50000,-,-,-,-,MDP20250109-163628-944363
2024/12/28 12:24:30,買い,15398762,BTC,0.0006494,-,JPY,-10000,自己,BFF20241228-032430-459608
2024/12/19 15:38:12,出金手数料,-,JPY,-770,-,-,-,-,MWD20241219-063812-636374F
2024/12/19 15:38:12,出金,-,JPY,-200000,-,-,-,-,MWD20241219-063812-636374
2024/10/16 19:27:25,売り,9811056,BTC,-0.01692323,-,JPY,166034,自己,BFF20241016-102725-859122
2024/10/16 17:27:26,売り,9734644,BTC,-0.00205452,-,JPY,20000,自己,BFF20241016-082726-402247
2024/10/15 23:28:04,売り,9702325,BTC,-0.00206137,-,JPY,20000,自己,BFF20241015-142804-999349
2024/10/15 07:30:23,売り,9587134,BTC,-0.00208613,-,JPY,20000,自己,BFF20241014-223023-136790
2024/08/07 04:08:32,入金,-,JPY,40000,-,-,-,-,MDP20240806-190832-587230
2024/08/06 15:47:23,買い,8387958,BTC,0.00417264,-,JPY,-35000,自己,BFF20240806-064723-754031
2024/08/06 13:46:50,買い,8312732,BTC,0.00180446,-,JPY,-15000,自己,BFF20240806-044650-949998
2024/08/06 00:38:59,入金,-,JPY,50000,-,-,-,-,MDP20240805-153859-036804
2024/08/05 23:37:54,買い,7908363,BTC,0.00189672,-,JPY,-15000,自己,BFF20240805-143754-118255
2024/08/05 17:56:21,買い,7727245,BTC,0.00452942,-,JPY,-35000,自己,BFF20240805-085621-944295
2024/08/05 16:34:03,入金,-,JPY,50000,-,-,-,-,MDP20240805-073403-664849
2024/08/03 07:32:49,買い,9358921,BTC,0.00534249,-,JPY,-50000,自己,BFF20240802-223249-815570
2024/08/03 07:31:10,入金,-,JPY,50000,-,-,-,-,MDP20240802-223110-790901
2024/08/02 01:16:46,買い,9772642,BTC,0.00511632,-,JPY,-50000,自己,BFF20240801-161646-599311
2024/08/02 01:08:11,入金,-,JPY,50000,-,-,-,-,MDP20240801-160811-471661
2024/08/01 18:51:36,買い,9977064,BTC,0.0002632,-,JPY,-2626,自己,BFF20240801-095136-854761
2024/08/01 18:49:09,売り,9392777,BTC,-0.00004,-,JPY,375,自己,BFF20240801-094909-816958
2024/08/01 17:47:03,買い,9971026,BTC,0.00000987,-,JPY,-99,自己,BFF20240801-084703-865893
2024/08/01 17:19:20,買い,9955635,BTC,0.00003003,-,JPY,-299,自己,BFF20240801-081920-442115
2024/08/01 17:18:23,買い,492282,ETH,0.0002011,-,JPY,-99,自己,BFF20240801-081823-427443
2024/08/01 17:18:09,買い,492191,ETH,0.00020317,-,JPY,-100,自己,BFF20240801-081809-424231
2024/08/01 17:13:36,買い,96.09,XRP,1.030284,-,JPY,-99,自己,BFF20240801-081336-354839
2024/08/01 17:11:26,買い,96.13,XRP,0.520128,-,JPY,-50,自己,BFF20240801-081126-322313
2024/08/01 17:03:03,買い,96.11,XRP,0.010404,-,JPY,-1,自己,BFF20240801-080303-194050
2024/08/01 17:00:29,買い,492665,ETH,0.00000202,-,JPY,-1,自己,BFF20240801-080029-154136
2024/08/01 17:00:07,買い,9954602,BTC,0.0000001,-,JPY,-1,自己,BFF20240801-080007-148842
2024/08/01 16:58:19,入金,-,JPY,3000,-,-,-,-,MDP20240801-075819-121947
""").strip()

COLS = ["取引日時","取引種別","価格","通貨1","数量1",
        "手数料","通貨2","数量2","区分","注文ID"]

fixed = pd.read_csv(io.StringIO(RAW), sep=",", header=0, names=COLS)
fixed["取引日時"] = pd.to_datetime(fixed["取引日時"],
                                  format="%Y/%m/%d %H:%M:%S")
for c in ["価格","数量1","手数料","数量2"]:
    fixed[c] = pd.to_numeric(fixed[c].replace("-", pd.NA), errors="coerce")
fixed["種別EN"] = fixed["取引種別"].map(JP2EN).fillna(fixed["取引種別"])
fixed["key"] = fixed.apply(
    lambda r: f"{r['取引日時']:%F %T}|{r['種別EN']}|{r['通貨1']}|{r['数量1']}",
    axis=1
)

# ---------- 3. API 側取得 & rows 組み立て ----------
def _api(path, params=None):
    ts = str(int(time.time()))
    q  = urlencode(params or {})
    target = f"{path}?{q}" if q else path
    sign = hmac.new(API_SECRET.encode(),(ts+"GET"+target).encode(),hashlib.sha256).hexdigest()
    hdr  = {"ACCESS-KEY":API_KEY,"ACCESS-TIMESTAMP":ts,"ACCESS-SIGN":sign}
    try:
        r = requests.get(API_URL+path, params=params, headers=hdr, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.warning("%s : %s", path, e)
        return []

rows, seen = [], set()

def add(row):
    tid = row[-1]
    if tid in seen: return
    seen.add(tid)
    rows.append(row)
# ---------- 4. DataFrame 化 & フィルタリング ----------
# API取得データをDataFrame化
api = pd.DataFrame(rows, columns=["取引日時","取引種別","価格","通貨1","数量1","注文ID"])
# 取引日時をdatetime型に変換
api["取引日時"] = pd.to_datetime(api["取引日時"])
api[["価格","数量1"]] = api[["価格","数量1"]].apply(pd.to_numeric, errors="coerce")
api["種別EN"] = api["取引種別"].map(JP2EN).fillna(api["取引種別"])
# Lightning子注文に紐づく「JPYの買い/売り」を除外
api = api[~((api["通貨1"] == "JPY") & api["取引種別"].isin(["買い","売り"]))]
# 重複キーで一意化
# 一意化キーをベクトル化で生成
api["key"] = api["取引日時"].dt.strftime("%F %T") + "|" + api["種別EN"] + "|" + api["通貨1"] + "|" + api["数量1"].astype(str)

api = api.drop_duplicates(subset="key")
# API側の子注文除外ロジック
api["ts_jst"] = api["取引日時"] + pd.Timedelta(hours=9)
api["ts_jst"] = api["ts_jst"].dt.floor("S")
# 同一日時・数量・売買のJPYレコードを抽出
jpy = api[(api["通貨1"] == "JPY") & api["取引種別"].isin(["買い","売り"])]
jpy_keys = set(zip(jpy["ts_jst"], jpy["取引種別"], jpy["数量1"]))
# 通貨がJPYでないエントリでJPYキーにマッチするものを除外
api = api[~((api["通貨1"] != "JPY") & api.apply(lambda r: (r["ts_jst"], r["取引種別"], r["数量1"]) in jpy_keys, axis=1))]

# ---------- 5. 突合 ----------
merged = fixed.merge(api[["key"]], on="key", how="outer", indicator=True)
print("\n=== 突合サマリ ===")
print(tabulate(merged["_merge"].value_counts().reset_index(), headers=["状態","件数"], tablefmt="github"))


# ---------- 6. Dash ダッシュボード ----------
def ticker_price(sym):
    if sym == "JPY": return 1.0
    try:
        return requests.get(API_URL+"/v1/ticker",
                            params={"product_code":f"{sym}_JPY"},
                            timeout=10).json()["ltp"]
    except:
        return 0.0

pos      = api.groupby("通貨1")["数量1"].sum()
total    = sum(pos[s]*ticker_price(s) for s in pos.index)
invested = -fixed.loc[fixed["通貨2"]=="JPY","数量2"].sum()
pnl      = total - invested

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.layout = dbc.Container([
    html.H4("bitFlyer Dashboard – FINAL"),
    html.P(f"評価額 {total:,.0f} JPY / 累積投資 {invested:,.0f} JPY / 損益 {pnl:+,.0f} JPY"),
    dash_table.DataTable(
        fixed.sort_values("取引日時"),
        page_size=25,
        style_cell={"fontFamily":"Menlo","fontSize":"11px"}
    )
], fluid=True)

if __name__ == "__main__":
    print("\nDash → http://127.0.0.1:8050/")  
