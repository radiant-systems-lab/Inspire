# Dataset Generation and Analysis
---
# Analysis Code
###segment_analysis.py
\<print params path\> \<GWL dir\> \<output.json\>  

Extracts all Write statements in the GWL files in "GWL dir". Analyzes the segments as if a representation of the design was constructed in memory. Calculates an adhesion score of each segment to its surroounding segments. Tags segments that are likely to cause failures in the print in the output json. Outputs JSON as a list of segments with start and end coordinates, per layer.

### blame_design.py

\<reduced.json\> \<segments\_analyzed.json\> [design.json]  
Iterates through error-tagged segments in segments_analysed.json. Geometrically computes mappings from segments onto nearest primitives in reduced.json (which must have hierarchy annotations - an addition from Simon's code), and calculates a breakdown of errors for each object. If a design.json is provided, outputs a new design.json where each component on each named object is annotated with a breakdown of errors contained therein. Otherwise, returns a JSON description of all errors per named object and component. This could be extended to work for designs and code other than the Named Object schema, given that segment_analysis could be run.

# Dataset Code
###dataset_generation.ipynb
A notebook that can generate the Dataset direcory. Processes and structures each design in the Dataset directory as exactly as would the output of NamedObjectAgent. Includes cells for generating the Qdrant datastore for vectorization of design prompts. The generated datastore can be used in NamedObjectAgent to improve accuracy by including examples in a RAG-style approach according to the design prompt, but falls back to a default example if not present. A language model is required to generate this datastore
