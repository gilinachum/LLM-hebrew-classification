# LLM-Hebrew-Classification-Benchmark
## Background
This benchmark contains code, dataset, and results of classifying Anthrpoic and OpenAI models over a use case for classifying customer inquiries to customer service in an imaginary financial institution.
## Getting started
First, make sure you have access to Anthropic models via Amazon Bedrock. OpenAI are accessed directly through openAI.  
`pip install -r requirements.txt`  
`python main.py --action evaluate --llm-names <model_name>`  
For example:  
`python main.py --action evaluate --llm-names anthropic.claude-instant-v1`  
`python main.py --action report --llm-names anthropic.claude-instant-v1`  

## Results
See [resuits.csv](./results.csv).  
See results per model in [detailed-results](./detailed-results).  
For more read this.

## License
Apache-2.0