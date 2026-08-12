"""Generate the seven small, executable teaching notebooks without hidden state."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"notebooks"

def write(name,title,cells):
    bootstrap = """# Make imports work whether VS Code starts in the repository or notebooks folder.
import os
import sys
from pathlib import Path

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if not (ROOT / "src").exists():
    raise RuntimeError(f"Cannot find the project root from {Path.cwd()}")
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("Project root:", ROOT)"""
    notebook=nbf.v4.new_notebook(); notebook.cells=[nbf.v4.new_markdown_cell(f"# {title}\nThe first cell automatically selects the repository root."),nbf.v4.new_code_cell(bootstrap)]+[nbf.v4.new_code_cell(cell) for cell in cells]
    notebook.metadata.kernelspec={"display_name":"Python 3 (SmartCharge)","language":"python","name":"python3"}; nbf.write(notebook,OUT/name)

write("01_dataset_analysis.ipynb","Dataset analysis",[
"import pandas as pd\nimport matplotlib.pyplot as plt\nfrom src.utils.config import project_path",
"path=project_path('data/raw/Indian_EV_Stations_Simplified.csv')\ndf=pd.read_csv(path)\nprint('Shape:',df.shape)\nprint('Columns:',df.columns.tolist())\ndisplay(df.head())",
"print('Missing values:\\n',df.isna().sum())\nprint('Duplicate rows:',df.duplicated().sum())",
"valid=df.Latitude.between(-90,90)&df.Longitude.between(-180,180)\nprint('Valid coordinates:',valid.sum())\ndf['Power (kW)'].hist(); plt.title('Charging power distribution'); plt.show()",
"display(df['Connector Type'].value_counts())\ndisplay(df.groupby('State').size().sort_values(ascending=False).head(15))",
"print('Dataset Validation Summary\\n--------------------------')\nprint('Rows:',len(df))\nprint('Valid coordinates:',valid.sum())\nprint('Duplicate stations:',df.duplicated(['Station Name','Latitude','Longitude']).sum())\nprint('Missing power:',df['Power (kW)'].isna().sum())\nprint('Connector types:',df['Connector Type'].nunique())\nprint('Power range:',df['Power (kW)'].min(),'-',df['Power (kW)'].max(),'kW')"])
write("02_environment_validation.ipynb","Environment validation",["from src.environment.ev_charging_env import EVChargingEnv\nfrom src.rl.common import validate_environment\nenv=EVChargingEnv(debug=True); validate_environment(env)","import numpy as np, pandas as pd\nenv.reset(seed=42); rows=[]\nfor step in range(50):\n a=int(env.np_random.choice(np.flatnonzero(env.candidate_mask))); _,reward,done,_,info=env.step(a); rows.append({'step':step,'reward':reward,'queue':sum(x.queue_length for x in env.queue_manager.stations.values()),'utilization':sum(x.utilization for x in env.queue_manager.stations.values())/len(env.queue_manager.stations)})\n if done: env.reset()\nframe=pd.DataFrame(rows); frame.set_index('step').plot(subplots=True,figsize=(10,8))"])
write("03_baseline_comparison.ipynb","Baseline comparison",["from src.evaluation.compare_models import compare\ntable=compare(); display(table)","table.set_index('model')['mean_reward'].plot.bar(title='Measured mean reward')"])
write("04_rl_model_comparison.ipynb","RL model comparison",["import pandas as pd\nfrom src.evaluation.train_and_compare import run_experiment\nfrom src.evaluation.compare_models import compare\nfrom src.utils.config import project_path\n# True retrains all three for equal 10,000-step budgets; False reevaluates saved models.\nRUN_TRAINING=False\nresults=run_experiment(timesteps=10_000,verbose=0) if RUN_TRAINING else compare()","import pandas as pd\nimport matplotlib.pyplot as plt\nfrom src.utils.config import project_path\ncolumns=['model','mean_reward','std_reward','success_rate','mean_total_time_minutes','mean_wait_minutes','mean_estimated_cost_inr','mean_utilization_imbalance','verdict']\ndisplay(results[columns].round(3))\nfigure,axes=plt.subplots(1,2,figsize=(14,5))\naxes[0].bar(results.model,results.mean_reward,yerr=results.std_reward,capsize=3); axes[0].set_title('Held-out reward: mean ± standard deviation'); axes[0].tick_params(axis='x',rotation=45)\nfor algorithm in ('ppo','dqn','a2c'):\n curve=pd.read_csv(project_path(f'results/metrics/{algorithm}_training_curve.csv')); axes[1].plot(curve.episode,curve.episode_reward.rolling(10,min_periods=1).mean(),label=algorithm.upper())\naxes[1].set_title('Training reward (10-episode rolling mean)'); axes[1].legend(); figure.tight_layout(); plt.show()"])
write("05_final_training.ipynb","Final training",["from src.rl.train_ppo import train\n# Increase config/default.yaml training budget for the final experiment.\nmodel=train('default')\nprint('Saved:', 'models/ppo_ev_charging.zip')"])
write("06_final_evaluation.ipynb","Final evaluation",["from src.evaluation.compare_models import compare\n# Extend evaluation seeds/config for peak, off-peak, rain, clear, demand and initial-congestion scenarios.\ntable=compare(); display(table)","print('Report mean ± standard deviation; do not reuse training seeds for the final report.')"])
write("07_review_demo.ipynb","One-click faculty review",["from src.utils.config import load_config\nfrom src.data.prepare_data import prepare_data\nconfig=load_config('review'); stations,report=prepare_data(config=config); display(report)","from src.network.station_graph import build_station_graph\ngraph=build_station_graph(stations.head(15),config['network']['edge_radius_km']); print(graph)","from src.environment.ev_charging_env import EVChargingEnv\nfrom src.baselines.policies import nearest_action,weighted_greedy_action\nenv=EVChargingEnv(stations,config,debug=True); observation,info=env.reset(seed=42); display(env.candidates)","recommendation=weighted_greedy_action(env); alternatives=[i for i,v in enumerate(env.candidate_mask) if v and i!=recommendation]; print('Recommended index:',recommendation,'alternatives:',alternatives[:3])","before=env.queue_manager.snapshot(); actual=alternatives[0] if alternatives else recommendation; env.set_actual_selection(actual); _,reward,_,_,result=env.step(recommendation); after=env.queue_manager.snapshot(); print('Recommended:',result['recommended_station_id']); print('Actual:',result['actual_selected_station_id']); print('Accepted:',result['recommendation_accepted']); print('Reward:',reward)","changed=[station for station in after if before[station]!=after[station]]; print('Changed stations:',changed); assert result['actual_selected_station_id'] in changed"])
print(f"Generated notebooks in {OUT}")
