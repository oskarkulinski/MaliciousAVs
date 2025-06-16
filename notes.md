Tests:
1. 17:00 12.06.2025 - 
    python scripts/ippo_torchrl.py --id saint_0 --alg-conf config5 --task-conf config5 --net saint_arnoult --env-seed 42 --torch-seed 0


python analysis/metrics.py --id <exp_id> --verbose True

## Meeting notes:  
20 40 60 avs%  
100 300 500 cars  
sumo-gui in simulator parameters  
tabular q learning aon.py example  
at least 500 days for tabular learning  
half of days to before epsilon zeroed  
obsrvations should be made into bins, maybe bins of 5 for 100 agents  
batch size 16-64 if deep learning  
config 1 if ippo 2 depth and less cells  
policy updates every day  


# How to run? 
Copy or link tabql.py into URB/baseline_models/  
Copy or link tabql_script.py into URB/scripts/  
Copy or link config_tab.json into URB/config/algo_config/baseline/  
Than run command:  
```python scripts/tabql.py --id <id> --alg-conf config_tab --task-conf config5 --net ingolstadt_custom --env-seed 42 --model tabql```  
After the experiment is complete run the analyisis script on the results.