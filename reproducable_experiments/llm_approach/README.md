# This folder contains agent-based LLM-based scripts

###dataset_evaluation.py
\<dataset_dir\>  
Evaluates the LLM agent, prompting the agent n (hardcoded) times against each item in the dataset directory. Calculates metrics for the average accuracy (a volume comparison of the similarity of the generated design schema to that in the dataset), and the consistency (a score based on the similarity of each generation). Produces json output in the current directory. _Note: this is not yet bug-free_

###NamedObjectAgent.py
Reads Docs/API.txt, prompt.txt in the current directory. Invokes an LLM to generate the design from prompt.txt. Runs all post-processing steps, including slicing, rendering, and segment_analysis.