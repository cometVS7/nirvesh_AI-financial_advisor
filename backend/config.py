from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "preprocessed"
    models_dir: Path = base_dir / "models"
    outputs_dir: Path = base_dir / "outputs"
    stock_cache_dir: Path = raw_dir / "stocks"
    news_cache_dir: Path = raw_dir / "news"
    policy_cache_dir: Path = raw_dir / "policy"
    regulatory_cache_dir: Path = raw_dir / "regulations"
    company_catalog_path: Path = data_dir / "company_catalog.csv"
    training_manifest_path: Path = processed_dir / "training_manifest.json"
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    news_api_key: str = os.getenv("NEWS_API_KEY", "")
    sebi_rss_url: str = os.getenv("SEBI_RSS_URL", "https://www.sebi.gov.in/sebirss.xml")
    upstox_access_token: str = os.getenv("UPSTOX_ACCESS_TOKEN", "")
    upstox_instruments_url: str = os.getenv("UPSTOX_INSTRUMENTS_URL", "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz")
    quote_proxy_url: str = os.getenv("QUOTE_PROXY_URL", "https://military-jobye-haiqstudios-14f59639.koyeb.app")
    backend_host: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    backend_port: int = int(os.getenv("BACKEND_PORT", "8000"))


settings = Settings()

HISTORICAL_REFRESH_HOURS = 24
NEWS_REFRESH_HOURS = 8
POLICY_REFRESH_DAYS = 7
REGULATION_REFRESH_DAYS = 15

SECTORS: List[str] = [
    "finance and banking",
    "automobile",
    "defence",
    "it and ai",
    "oil and gas and energy",
    "fmcg",
    "healthcare and pharma",
    "textile",
    "agriculture",
]

SECTOR_LEADERS: Dict[str, List[Dict[str, str]]] = {
    "finance and banking": [
        {"company": "HDFC Bank", "ticker": "HDFCBANK.NS"},
        {"company": "ICICI Bank", "ticker": "ICICIBANK.NS"},
        {"company": "State Bank of India", "ticker": "SBIN.NS"},
        {"company": "Axis Bank", "ticker": "AXISBANK.NS"},
        {"company": "Kotak Mahindra Bank", "ticker": "KOTAKBANK.NS"},
        {"company": "Bajaj Finance", "ticker": "BAJFINANCE.NS"},
    ],
    "automobile": [
        {"company": "Maruti Suzuki", "ticker": "MARUTI.NS"},
        {"company": "Tata Motors Passenger Vehicles", "ticker": "TMPV.NS"},
        {"company": "Mahindra and Mahindra", "ticker": "M&M.NS"},
        {"company": "Bajaj Auto", "ticker": "BAJAJ-AUTO.NS"},
        {"company": "Eicher Motors", "ticker": "EICHERMOT.NS"},
        {"company": "Hero MotoCorp", "ticker": "HEROMOTOCO.NS"},
    ],
    "defence": [
        {"company": "Hindustan Aeronautics", "ticker": "HAL.NS"},
        {"company": "Bharat Electronics", "ticker": "BEL.NS"},
        {"company": "Bharat Dynamics", "ticker": "BDL.NS"},
        {"company": "Mazagon Dock", "ticker": "MAZDOCK.NS"},
        {"company": "Cochin Shipyard", "ticker": "COCHINSHIP.NS"},
        {"company": "Data Patterns", "ticker": "DATAPATTNS.NS"},
    ],
    "it and ai": [
        {"company": "TCS", "ticker": "TCS.NS"},
        {"company": "Infosys", "ticker": "INFY.NS"},
        {"company": "HCL Technologies", "ticker": "HCLTECH.NS"},
        {"company": "Wipro", "ticker": "WIPRO.NS"},
        {"company": "Tech Mahindra", "ticker": "TECHM.NS"},
        {"company": "Persistent Systems", "ticker": "PERSISTENT.NS"},
    ],
    "oil and gas and energy": [
        {"company": "Reliance Industries", "ticker": "RELIANCE.NS"},
        {"company": "ONGC", "ticker": "ONGC.NS"},
        {"company": "Indian Oil", "ticker": "IOC.NS"},
        {"company": "NTPC", "ticker": "NTPC.NS"},
        {"company": "Power Grid", "ticker": "POWERGRID.NS"},
        {"company": "Coal India", "ticker": "COALINDIA.NS"},
    ],
    "fmcg": [
        {"company": "Hindustan Unilever", "ticker": "HINDUNILVR.NS"},
        {"company": "ITC", "ticker": "ITC.NS"},
        {"company": "Nestle India", "ticker": "NESTLEIND.NS"},
        {"company": "Britannia", "ticker": "BRITANNIA.NS"},
        {"company": "Dabur", "ticker": "DABUR.NS"},
        {"company": "Godrej Consumer", "ticker": "GODREJCP.NS"},
    ],
    "healthcare and pharma": [
        {"company": "Sun Pharma", "ticker": "SUNPHARMA.NS"},
        {"company": "Dr Reddy's", "ticker": "DRREDDY.NS"},
        {"company": "Cipla", "ticker": "CIPLA.NS"},
        {"company": "Divi's Laboratories", "ticker": "DIVISLAB.NS"},
        {"company": "Apollo Hospitals", "ticker": "APOLLOHOSP.NS"},
        {"company": "Lupin", "ticker": "LUPIN.NS"},
    ],
    "textile": [
        {"company": "KPR Mill", "ticker": "KPRMILL.NS"},
        {"company": "Trident", "ticker": "TRIDENT.NS"},
        {"company": "Arvind", "ticker": "ARVIND.NS"},
        {"company": "Welspun Living", "ticker": "WELSPUNLIV.NS"},
        {"company": "Vardhman Textiles", "ticker": "VTL.NS"},
        {"company": "Raymond Lifestyle", "ticker": "RAYMONDLSL.NS"},
    ],
    "agriculture": [
        {"company": "UPL", "ticker": "UPL.NS"},
        {"company": "PI Industries", "ticker": "PIIND.NS"},
        {"company": "Coromandel International", "ticker": "COROMANDEL.NS"},
        {"company": "Rallis India", "ticker": "RALLIS.NS"},
        {"company": "Chambal Fertilisers", "ticker": "CHAMBLFERT.NS"},
        {"company": "Godrej Agrovet", "ticker": "GODREJAGRO.NS"},
    ],
}

REAL_STOCK_UNIVERSE: Dict[str, List[Dict[str, str]]] = {
    "finance and banking": [
        {"company": "HDFC Bank", "ticker": "HDFCBANK.NS"},
        {"company": "ICICI Bank", "ticker": "ICICIBANK.NS"},
        {"company": "State Bank of India", "ticker": "SBIN.NS"},
        {"company": "Axis Bank", "ticker": "AXISBANK.NS"},
        {"company": "Kotak Mahindra Bank", "ticker": "KOTAKBANK.NS"},
        {"company": "Bajaj Finance", "ticker": "BAJFINANCE.NS"},
        {"company": "IndusInd Bank", "ticker": "INDUSINDBK.NS"},
        {"company": "Bank of Baroda", "ticker": "BANKBARODA.NS"},
        {"company": "Punjab National Bank", "ticker": "PNB.NS"},
        {"company": "Federal Bank", "ticker": "FEDERALBNK.NS"},
        {"company": "IDFC First Bank", "ticker": "IDFCFIRSTB.NS"},
        {"company": "AU Small Finance Bank", "ticker": "AUBANK.NS"},
        {"company": "Canara Bank", "ticker": "CANBK.NS"},
        {"company": "Union Bank of India", "ticker": "UNIONBANK.NS"},
        {"company": "Indian Bank", "ticker": "INDIANB.NS"},
        {"company": "Yes Bank", "ticker": "YESBANK.NS"},
        {"company": "Bandhan Bank", "ticker": "BANDHANBNK.NS"},
        {"company": "RBL Bank", "ticker": "RBLBANK.NS"},
        {"company": "Karur Vysya Bank", "ticker": "KARURVYSYA.NS"},
        {"company": "South Indian Bank", "ticker": "SOUTHBANK.NS"},
        {"company": "City Union Bank", "ticker": "CUB.NS"},
        {"company": "IDBI Bank", "ticker": "IDBI.NS"},
        {"company": "LIC Housing Finance", "ticker": "LICHSGFIN.NS"},
        {"company": "Cholamandalam Investment", "ticker": "CHOLAFIN.NS"},
        {"company": "Muthoot Finance", "ticker": "MUTHOOTFIN.NS"},
        # UPDATED: Piramal Enterprises merged into Piramal Finance
        {"company": "Piramal Finance", "ticker": "PIRAMALFIN.NS"},
        {"company": "Shriram Finance", "ticker": "SHRIRAMFIN.NS"},
        {"company": "SBI Cards", "ticker": "SBICARD.NS"},
        {"company": "Rec", "ticker": "RECLTD.NS"},
        {"company": "Power Finance Corporation", "ticker": "PFC.NS"},
    ],
    "automobile": [
        {"company": "Maruti Suzuki", "ticker": "MARUTI.NS"},
        {"company": "Tata Motors Commercial Vehicles", "ticker": "TMCV.NS"},
        {"company": "Tata Motors Passenger Vehicles", "ticker": "TMPV.NS"},
        {"company": "Mahindra and Mahindra", "ticker": "M&M.NS"},
        {"company": "Bajaj Auto", "ticker": "BAJAJ-AUTO.NS"},
        {"company": "Eicher Motors", "ticker": "EICHERMOT.NS"},
        {"company": "Hero MotoCorp", "ticker": "HEROMOTOCO.NS"},
        {"company": "TVS Motor", "ticker": "TVSMOTOR.NS"},
        {"company": "Ashok Leyland", "ticker": "ASHOKLEY.NS"},
        {"company": "Samvardhana Motherson", "ticker": "MOTHERSON.NS"},
        {"company": "Bosch", "ticker": "BOSCHLTD.NS"},
        {"company": "Balkrishna Industries", "ticker": "BALKRISIND.NS"},
        {"company": "MRF", "ticker": "MRF.NS"},
        {"company": "Apollo Tyres", "ticker": "APOLLOTYRE.NS"},
        {"company": "CEAT", "ticker": "CEATLTD.NS"},
        {"company": "Exide Industries", "ticker": "EXIDEIND.NS"},
        {"company": "Amara Raja Energy & Mobility", "ticker": "ARE&M.NS"},
        {"company": "Sona BLW Precision", "ticker": "SONACOMS.NS"},
        {"company": "Schaeffler India", "ticker": "SCHAEFFLER.NS"},
        {"company": "Endurance Technologies", "ticker": "ENDURANCE.NS"},
        {"company": "UNO Minda", "ticker": "UNOMINDA.NS"},
        {"company": "Minda Corporation", "ticker": "MINDACORP.NS"},
        {"company": "Jamna Auto Industries", "ticker": "JAMNAAUTO.NS"},
        {"company": "Craftsman Automation", "ticker": "CRAFTSMAN.NS"},
        {"company": "JK Tyre", "ticker": "JKTYRE.NS"},
        {"company": "Suprajit Engineering", "ticker": "SUPRAJIT.NS"},
        {"company": "Varroc Engineering", "ticker": "VARROC.NS"},
        {"company": "Gabriel India", "ticker": "GABRIEL.NS"},
        {"company": "Pricol", "ticker": "PRICOLLTD.NS"},
        {"company": "Force Motors", "ticker": "FORCEMOT.NS"},
        {"company": "Olectra Greentech", "ticker": "OLECTRA.NS"},
    ],
    "defence": [
        {"company": "Hindustan Aeronautics", "ticker": "HAL.NS"},
        {"company": "Bharat Electronics", "ticker": "BEL.NS"},
        {"company": "Bharat Dynamics", "ticker": "BDL.NS"},
        {"company": "Mazagon Dock", "ticker": "MAZDOCK.NS"},
        {"company": "Cochin Shipyard", "ticker": "COCHINSHIP.NS"},
        {"company": "Data Patterns", "ticker": "DATAPATTNS.NS"},
        {"company": "Paras Defence", "ticker": "PARAS.NS"},
        {"company": "Astra Microwave", "ticker": "ASTRAMICRO.NS"},
        {"company": "BEML", "ticker": "BEML.NS"},
        {"company": "Garden Reach Shipbuilders", "ticker": "GRSE.NS"},
        {"company": "MTAR Technologies", "ticker": "MTARTECH.NS"},
        {"company": "ideaForge Technology", "ticker": "IDEAFORGE.NS"},
        {"company": "Taneja Aerospace", "ticker": "TAALENT.NS"},
        {"company": "Premier Explosives", "ticker": "PREMEXPLN.NS"},
        {"company": "Mishra Dhatu Nigam", "ticker": "MIDHANI.NS"},
        {"company": "Solar Industries", "ticker": "SOLARINDS.NS"},
        {"company": "Cyient DLM", "ticker": "CYIENTDLM.NS"},
        {"company": "Unimech Aerospace", "ticker": "UNIMECH.NS"},
        {"company": "Zen Technologies", "ticker": "ZENTEC.NS"},
        {"company": "Apollo Micro Systems", "ticker": "APOLLO.NS"},
        {"company": "Tata Power", "ticker": "TATAPOWER.NS"},
        {"company": "Larsen and Toubro", "ticker": "LT.NS"},
        {"company": "Bharat Forge", "ticker": "BHARATFORG.NS"},
        {"company": "DCX Systems", "ticker": "DCXINDIA.NS"},
        {"company": "Sika Interplant", "ticker": "SIKA.NS"},
        {"company": "Krishna Defence", "ticker": "KRISHNADEF.NS"},
        {"company": "Rossell Techsys", "ticker": "ROSSTECH.NS"},
        # UPDATED: Aeron Composites migrated to mainboard
        {"company": "Aeron Composites", "ticker": "AERON.NS"},
        {"company": "TechEra Engineering", "ticker": "TECHERA.NS"},
        {"company": "VEM Technologies", "ticker": "VEMTECH.NS"},
    ],
    "it and ai": [
        {"company": "TCS", "ticker": "TCS.NS"},
        {"company": "Infosys", "ticker": "INFY.NS"},
        {"company": "HCL Technologies", "ticker": "HCLTECH.NS"},
        {"company": "Wipro", "ticker": "WIPRO.NS"},
        {"company": "Tech Mahindra", "ticker": "TECHM.NS"},
        {"company": "Persistent Systems", "ticker": "PERSISTENT.NS"},
        {"company": "LTIMindtree", "ticker": "LTIM.NS"},
        {"company": "Mphasis", "ticker": "MPHASIS.NS"},
        {"company": "Coforge", "ticker": "COFORGE.NS"},
        {"company": "Oracle Financial Services", "ticker": "OFSS.NS"},
        {"company": "KPIT Technologies", "ticker": "KPITTECH.NS"},
        {"company": "Birlasoft", "ticker": "BSOFT.NS"},
        {"company": "Tata Elxsi", "ticker": "TATAELXSI.NS"},
        {"company": "Cyient", "ticker": "CYIENT.NS"},
        {"company": "Sonata Software", "ticker": "SONATSOFTW.NS"},
        {"company": "Zensar Technologies", "ticker": "ZENSARTECH.NS"},
        {"company": "Happiest Minds", "ticker": "HAPPSTMNDS.NS"},
        {"company": "Tanla Platforms", "ticker": "TANLA.NS"},
        {"company": "Intellect Design", "ticker": "INTELLECT.NS"},
        {"company": "Firstsource Solutions", "ticker": "FSL.NS"},
        {"company": "Nucleus Software", "ticker": "NUCLEUS.NS"},
        {"company": "eClerx Services", "ticker": "ECLERX.NS"},
        {"company": "Syrma SGS", "ticker": "SYRMA.NS"},
        {"company": "Route Mobile", "ticker": "ROUTE.NS"},
        {"company": "Affle India", "ticker": "AFFLE.NS"},
        {"company": "Newgen Software", "ticker": "NEWGEN.NS"},
        {"company": "Saksoft", "ticker": "SAKSOFT.NS"},
        {"company": "Datamatics", "ticker": "DATAMATICS.NS"},
        {"company": "RateGain Travel", "ticker": "RATEGAIN.NS"},
        {"company": "Mastek", "ticker": "MASTEK.NS"},
    ],
    "oil and gas and energy": [
        {"company": "Reliance Industries", "ticker": "RELIANCE.NS"},
        {"company": "ONGC", "ticker": "ONGC.NS"},
        {"company": "Indian Oil", "ticker": "IOC.NS"},
        {"company": "NTPC", "ticker": "NTPC.NS"},
        {"company": "Power Grid", "ticker": "POWERGRID.NS"},
        {"company": "Coal India", "ticker": "COALINDIA.NS"},
        {"company": "Bharat Petroleum", "ticker": "BPCL.NS"},
        {"company": "Hindustan Petroleum", "ticker": "HINDPETRO.NS"},
        {"company": "GAIL", "ticker": "GAIL.NS"},
        {"company": "Oil India", "ticker": "OIL.NS"},
        {"company": "Adani Green Energy", "ticker": "ADANIGREEN.NS"},
        {"company": "Adani Energy Solutions", "ticker": "ADANIENSOL.NS"},
        {"company": "Torrent Power", "ticker": "TORNTPOWER.NS"},
        {"company": "Tata Power", "ticker": "TATAPOWER.NS"},
        {"company": "JSW Energy", "ticker": "JSWENERGY.NS"},
        {"company": "SJVN", "ticker": "SJVN.NS"},
        {"company": "NHPC", "ticker": "NHPC.NS"},
        {"company": "NLC India", "ticker": "NLCINDIA.NS"},
        {"company": "Petronet LNG", "ticker": "PETRONET.NS"},
        {"company": "Gujarat Gas", "ticker": "GUJGASLTD.NS"},
        {"company": "Mahanagar Gas", "ticker": "MGL.NS"},
        {"company": "Indraprastha Gas", "ticker": "IGL.NS"},
        {"company": "Solar Industries", "ticker": "SOLARINDS.NS"},
        {"company": "Waaree Energies", "ticker": "WAAREEENER.NS"},
        {"company": "Inox Wind", "ticker": "INOXWIND.NS"},
        {"company": "Suzlon Energy", "ticker": "SUZLON.NS"},
        {"company": "CESC", "ticker": "CESC.NS"},
        {"company": "KPI Green Energy", "ticker": "KPIGREEN.NS"},
        {"company": "Borosil Renewables", "ticker": "BORORENEW.NS"},
        {"company": "IEX", "ticker": "IEX.NS"},
    ],
    "fmcg": [
        {"company": "Hindustan Unilever", "ticker": "HINDUNILVR.NS"},
        {"company": "ITC", "ticker": "ITC.NS"},
        {"company": "Nestle India", "ticker": "NESTLEIND.NS"},
        {"company": "Britannia", "ticker": "BRITANNIA.NS"},
        {"company": "Dabur", "ticker": "DABUR.NS"},
        {"company": "Godrej Consumer", "ticker": "GODREJCP.NS"},
        {"company": "Tata Consumer", "ticker": "TATACONSUM.NS"},
        {"company": "Marico", "ticker": "MARICO.NS"},
        {"company": "Colgate Palmolive India", "ticker": "COLPAL.NS"},
        {"company": "UBL", "ticker": "UBL.NS"},
        {"company": "United Spirits", "ticker": "UNITDSPR.NS"},
        {"company": "Radico Khaitan", "ticker": "RADICO.NS"},
        {"company": "Emami", "ticker": "EMAMILTD.NS"},
        {"company": "Jyothy Labs", "ticker": "JYOTHYLAB.NS"},
        {"company": "Gillette India", "ticker": "GILLETTE.NS"},
        {"company": "Varun Beverages", "ticker": "VBL.NS"},
        {"company": "Bikaji Foods", "ticker": "BIKAJI.NS"},
        {"company": "Balrampur Chini", "ticker": "BALRAMCHIN.NS"},
        {"company": "KRBL", "ticker": "KRBL.NS"},
        {"company": "Bajaj Consumer Care", "ticker": "BAJAJCON.NS"},
        {"company": "Mrs Bectors Food", "ticker": "BECTORFOOD.NS"},
        # UPDATED: Adani Wilmar rebranded to AWL Agri Business
        {"company": "AWL Agri Business", "ticker": "AWL.NS"},
        {"company": "Zydus Wellness", "ticker": "ZYDUSWELL.NS"},
        {"company": "TTK Prestige", "ticker": "TTKPRESTIG.NS"},
        {"company": "Crompton Consumer", "ticker": "CROMPTON.NS"},
        {"company": "Patanjali Foods", "ticker": "PATANJALI.NS"},
        {"company": "Heritage Foods", "ticker": "HERITGFOOD.NS"},
        {"company": "Sula Vineyards", "ticker": "SULA.NS"},
        {"company": "Eveready Industries", "ticker": "EVEREADY.NS"},
        {"company": "VIP Industries", "ticker": "VIPIND.NS"},
    ],
    "healthcare and pharma": [
        {"company": "Sun Pharma", "ticker": "SUNPHARMA.NS"},
        {"company": "Dr Reddy's", "ticker": "DRREDDY.NS"},
        {"company": "Cipla", "ticker": "CIPLA.NS"},
        {"company": "Divi's Laboratories", "ticker": "DIVISLAB.NS"},
        {"company": "Apollo Hospitals", "ticker": "APOLLOHOSP.NS"},
        {"company": "Lupin", "ticker": "LUPIN.NS"},
        {"company": "Aurobindo Pharma", "ticker": "AUROPHARMA.NS"},
        {"company": "Torrent Pharmaceuticals", "ticker": "TORNTPHARM.NS"},
        {"company": "Mankind Pharma", "ticker": "MANKIND.NS"},
        {"company": "Alkem Laboratories", "ticker": "ALKEM.NS"},
        {"company": "Biocon", "ticker": "BIOCON.NS"},
        {"company": "Max Healthcare", "ticker": "MAXHEALTH.NS"},
        {"company": "Fortis Healthcare", "ticker": "FORTIS.NS"},
        {"company": "Glenmark", "ticker": "GLENMARK.NS"},
        {"company": "Abbott India", "ticker": "ABBOTINDIA.NS"},
        {"company": "Zydus Lifesciences", "ticker": "ZYDUSLIFE.NS"},
        {"company": "JB Chemicals", "ticker": "JBCHEPHARM.NS"},
        {"company": "Ajanta Pharma", "ticker": "AJANTPHARM.NS"},
        {"company": "Ipca Laboratories", "ticker": "IPCALAB.NS"},
        {"company": "Syngene International", "ticker": "SYNGENE.NS"},
        {"company": "Granules India", "ticker": "GRANULES.NS"},
        {"company": "Pfizer India", "ticker": "PFIZER.NS"},
        {"company": "Aster DM Healthcare", "ticker": "ASTERDM.NS"},
        {"company": "Narayana Hrudayalaya", "ticker": "NH.NS"},
        {"company": "Metropolis Healthcare", "ticker": "METROPOLIS.NS"},
        {"company": "Laurus Labs", "ticker": "LAURUSLABS.NS"},
        {"company": "Caplin Point", "ticker": "CAPLIPOINT.NS"},
        {"company": "Krishna Institute", "ticker": "KIMS.NS"},
        {"company": "Rainbow Childrens", "ticker": "RAINBOW.NS"},
        {"company": "Shilpa Medicare", "ticker": "SHILPAMED.NS"},
    ],
    "textile": [
        {"company": "KPR Mill", "ticker": "KPRMILL.NS"},
        {"company": "Trident", "ticker": "TRIDENT.NS"},
        {"company": "Arvind", "ticker": "ARVIND.NS"},
        {"company": "Welspun Living", "ticker": "WELSPUNLIV.NS"},
        {"company": "Vardhman Textiles", "ticker": "VTL.NS"},
        {"company": "Raymond Lifestyle", "ticker": "RAYMONDLSL.NS"},
        {"company": "Page Industries", "ticker": "PAGEIND.NS"},
        {"company": "Aditya Birla Fashion", "ticker": "ABFRL.NS"},
        {"company": "Lux Industries", "ticker": "LUXIND.NS"},
        {"company": "Rupa", "ticker": "RUPA.NS"},
        {"company": "Dollar Industries", "ticker": "DOLLAR.NS"},
        {"company": "Kitex Garments", "ticker": "KITEX.NS"},
        {"company": "Alok Industries", "ticker": "ALOKINDS.NS"},
        {"company": "Gokaldas Exports", "ticker": "GOKEX.NS"},
        {"company": "Sutlej Textiles", "ticker": "SUTLEJTEX.NS"},
        {"company": "Nitin Spinners", "ticker": "NITINSPIN.NS"},
        {"company": "LS Industries", "ticker": "LSIND.NS"},
        {"company": "Swan Energy", "ticker": "SWANENERGY.NS"},
        {"company": "Garware Technical Fibres", "ticker": "GARFIBRES.NS"},
        {"company": "Siyaram Silk Mills", "ticker": "SIYSIL.NS"},
        {"company": "Indo Count", "ticker": "ICIL.NS"},
        {"company": "Pearl Global", "ticker": "PGIL.NS"},
        {"company": "Monte Carlo", "ticker": "MONTECARLO.NS"},
        {"company": "Donear Industries", "ticker": "DONEAR.NS"},
        {"company": "RSWM", "ticker": "RSWM.NS"},
        {"company": "Himatsingka Seide", "ticker": "HIMATSEIDE.NS"},
        {"company": "Sarla Performance", "ticker": "SARLAPOLY.NS"},
        {"company": "Sportking India", "ticker": "SPORTKING.NS"},
        {"company": "Filatex India", "ticker": "FILATEX.NS"},
        {"company": "Cantabil Retail", "ticker": "CANTABIL.NS"},
    ],
    "agriculture": [
        {"company": "UPL", "ticker": "UPL.NS"},
        {"company": "PI Industries", "ticker": "PIIND.NS"},
        {"company": "Coromandel International", "ticker": "COROMANDEL.NS"},
        {"company": "Rallis India", "ticker": "RALLIS.NS"},
        {"company": "Chambal Fertilisers", "ticker": "CHAMBLFERT.NS"},
        {"company": "Godrej Agrovet", "ticker": "GODREJAGRO.NS"},
        {"company": "Gujarat State Fertilizers", "ticker": "GSFC.NS"},
        {"company": "Gujarat Narmada Valley Fert", "ticker": "GNFC.NS"},
        {"company": "Deepak Fertilisers", "ticker": "DEEPAKFERT.NS"},
        {"company": "National Fertilizers", "ticker": "NFL.NS"},
        {"company": "Rashtriya Chemicals", "ticker": "RCF.NS"},
        {"company": "Paradeep Phosphates", "ticker": "PARADEEP.NS"},
        {"company": "Kaveri Seed", "ticker": "KSCL.NS"},
        {"company": "Dhanuka Agritech", "ticker": "DHANUKA.NS"},
        {"company": "Sharda Cropchem", "ticker": "SHARDACROP.NS"},
        {"company": "Bayer Cropsciences", "ticker": "BAYERCROP.NS"},
        {"company": "Insecticides India", "ticker": "INSECTICID.NS"},
        {"company": "Best Agrolife", "ticker": "BESTAGRO.NS"},
        {"company": "Jubilant Ingrevia", "ticker": "JUBLINGREA.NS"},
        {"company": "Fertilizers and Chemicals Travancore", "ticker": "FACT.NS"},
        {"company": "DCM Shriram", "ticker": "DCMSHRIRAM.NS"},
        {"company": "Avanti Feeds", "ticker": "AVANTIFEED.NS"},
        {"company": "Astec LifeSciences", "ticker": "ASTEC.NS"},
        {"company": "Heranba Industries", "ticker": "HERANBA.NS"},
        {"company": "Mangalam Seeds", "ticker": "MANGALAM.NS"},
        {"company": "NACL Industries", "ticker": "NACLIND.NS"},
        {"company": "JK Agri Genetics", "ticker": "JKAGRI.NS"},
        {"company": "ARCL Organics", "ticker": "ARCL.NS"},
        {"company": "Agro Tech Foods", "ticker": "ATFL.NS"},
        {"company": "Vikas Ecotech", "ticker": "VIKASECO.NS"},
    ],
}

RISK_MAP = {"low": 0.92, "mid": 1.0, "high": 1.08}
HOLDING_WINDOWS = {"short term": 15, "medium term": 30, "long term": 90}

HISTORICAL_TICKER_ALIASES: Dict[str, List[str]] = {
    "ADANIENSOL.NS": ["ADANITRANS.NS"],
    "ARE&M.NS": ["AMARAJABAT.NS"],
    "TMPV.NS": ["TATAMOTORS.NS"],
    "RAYMONDLSL.NS": ["RAYMOND.NS"],
}


def ensure_directories() -> None:
    for path in [
        settings.data_dir,
        settings.raw_dir,
        settings.processed_dir,
        settings.models_dir,
        settings.outputs_dir,
        settings.stock_cache_dir,
        settings.news_cache_dir,
        settings.policy_cache_dir,
        settings.regulatory_cache_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def build_company_catalog(companies_per_sector: int = 30) -> pd.DataFrame:
    ensure_directories()
    if settings.company_catalog_path.exists():
        cached = pd.read_csv(settings.company_catalog_path)
        sector_counts = cached.groupby("sector").size().to_dict()
        enough_real = all(sector_counts.get(sector, 0) >= companies_per_sector for sector in SECTORS)
        if enough_real:
            return cached

    records = []
    leader_tickers = {sector: {item["ticker"] for item in leaders} for sector, leaders in SECTOR_LEADERS.items()}
    for sector in SECTORS:
        real_companies = REAL_STOCK_UNIVERSE.get(sector, [])[:companies_per_sector]
        for item in real_companies:
            records.append(
                {
                    "company": item["company"],
                    "ticker": item["ticker"],
                    "sector": sector,
                    "is_leader": item["ticker"] in leader_tickers.get(sector, set()),
                }
            )

    df = pd.DataFrame(records).drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    df.to_csv(settings.company_catalog_path, index=False)
    return df
