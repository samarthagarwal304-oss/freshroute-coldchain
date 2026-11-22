PY ?= python3
export PYTHONPATH := src

.PHONY: setup data train evalset eval eval-llm test serve demo smoke all clean
setup:        ## install dependencies
	$(PY) -m pip install -r requirements.txt
data:         ## generate synthetic sensors + trips (~20 s)
	$(PY) -m freshroute.datagen
train:        ## features, time-split model, metrics vs baseline, data-quality note (~25 s)
	$(PY) -m freshroute.train
	$(PY) -m freshroute.eda
evalset:      ## (re)build eval/questions.json from the generated data
	$(PY) eval/make_eval_set.py
eval:         ## score the assistant offline (no API key needed)
	$(PY) eval/run_eval.py --backend offline
eval-llm:     ## score the assistant with real Claude (needs ANTHROPIC_API_KEY)
	$(PY) eval/run_eval.py --backend anthropic
test:         ## unit + integration tests
	$(PY) -m pytest -q
serve:        ## API on http://localhost:8000/docs
	$(PY) -m uvicorn freshroute.api:app --port 8000
demo:         ## Streamlit UI
	$(PY) -m streamlit run app/streamlit_app.py
smoke:        ## start API, curl every endpoint, stop API
	bash scripts/smoke_test.sh
all: data train evalset eval test
