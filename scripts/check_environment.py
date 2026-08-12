import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.environment.ev_charging_env import EVChargingEnv
from src.rl.common import validate_environment
if __name__=="__main__": validate_environment(EVChargingEnv())
