python scripts/make_configs.py --list
16 runs:

  llama8b_scifact_top20_paper                  -> configs/llama8b_scifact_top20_paper.yaml
  llama8b_scifact_top20_compact                -> configs/llama8b_scifact_top20_compact.yaml
  qwen7b_scifact_top20_paper                   -> configs/qwen7b_scifact_top20_paper.yaml
  qwen7b_scifact_top20_compact                 -> configs/qwen7b_scifact_top20_compact.yaml
  llama8b_nfcorpus_top20_paper                 -> configs/llama8b_nfcorpus_top20_paper.yaml
  llama8b_nfcorpus_top20_compact               -> configs/llama8b_nfcorpus_top20_compact.yaml
  qwen7b_nfcorpus_top20_paper                  -> configs/qwen7b_nfcorpus_top20_paper.yaml
  qwen7b_nfcorpus_top20_compact                -> configs/qwen7b_nfcorpus_top20_compact.yaml
  llama8b_covid_top20_paper                    -> configs/llama8b_covid_top20_paper.yaml
  llama8b_covid_top20_compact                  -> configs/llama8b_covid_top20_compact.yaml
  qwen7b_covid_top20_paper                     -> configs/qwen7b_covid_top20_paper.yaml
  qwen7b_covid_top20_compact                   -> configs/qwen7b_covid_top20_compact.yaml
  llama8b_dbpedia_top20_paper                  -> configs/llama8b_dbpedia_top20_paper.yaml
  llama8b_dbpedia_top20_compact                -> configs/llama8b_dbpedia_top20_compact.yaml
  qwen7b_dbpedia_top20_paper                   -> configs/qwen7b_dbpedia_top20_paper.yaml
  qwen7b_dbpedia_top20_compact                 -> configs/qwen7b_dbpedia_top20_compact.yaml

sbatch lines:

D sbatch --job-name=llama8b_scifact_top20_paper slurm/pg.slurm configs/llama8b_scifact_top20_paper.yaml
D sbatch --job-name=llama8b_scifact_top20_compact slurm/pg.slurm configs/llama8b_scifact_top20_compact.yaml
D sbatch --job-name=qwen7b_scifact_top20_paper slurm/pg.slurm configs/qwen7b_scifact_top20_paper.yaml
D sbatch --job-name=qwen7b_scifact_top20_compact slurm/pg.slurm configs/qwen7b_scifact_top20_compact.yaml
D sbatch --job-name=llama8b_nfcorpus_top20_paper slurm/pg.slurm configs/llama8b_nfcorpus_top20_paper.yaml
D sbatch --job-name=llama8b_nfcorpus_top20_compact slurm/pg.slurm configs/llama8b_nfcorpus_top20_compact.yaml
D sbatch --job-name=qwen7b_nfcorpus_top20_paper slurm/pg.slurm configs/qwen7b_nfcorpus_top20_paper.yaml
D sbatch --job-name=qwen7b_nfcorpus_top20_compact slurm/pg.slurm configs/qwen7b_nfcorpus_top20_compact.yaml
D sbatch --job-name=llama8b_covid_top20_paper slurm/pg.slurm configs/llama8b_covid_top20_paper.yaml
D sbatch --job-name=llama8b_covid_top20_compact slurm/pg.slurm configs/llama8b_covid_top20_compact.yaml
D sbatch --job-name=qwen7b_covid_top20_paper slurm/pg.slurm configs/qwen7b_covid_top20_paper.yaml
D sbatch --job-name=qwen7b_covid_top20_compact slurm/pg.slurm configs/qwen7b_covid_top20_compact.yaml
D sbatch --job-name=llama8b_dbpedia_top20_paper slurm/pg.slurm configs/llama8b_dbpedia_top20_paper.yaml
D sbatch --job-name=llama8b_dbpedia_top20_compact slurm/pg.slurm configs/llama8b_dbpedia_top20_compact.yaml
D sbatch --job-name=qwen7b_dbpedia_top20_paper slurm/pg.slurm configs/qwen7b_dbpedia_top20_paper.yaml
D sbatch --job-name=qwen7b_dbpedia_top20_compact slurm/pg.slurm configs/qwen7b_dbpedia_top20_compact.yaml
