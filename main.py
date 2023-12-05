import openai
import os
import API_Keys
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.llms import OpenAI  
from langchain.llms import AI21
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#logging.getLogger().setLevel(logging.INFO)

def get_llm(llm_name):
    import os
    os.environ["OPENAI_API_KEY"] = API_Keys.OPENAI_API_KEY
    os.environ["AI21_API_KEY"] = API_Keys.AI21_API_KEY
    # Main LLM
    if (llm_name.startswith("j2")):
        llm = AI21(model=llm_name, temperature=0)
        return llm
    
    if (llm_name.startswith("text-davinci")):
        llm = OpenAI(model=llm_name, temperature=0,max_tokens=50)
        return llm
    
    if (llm_name.startswith("gpt-")):
        llm = ChatOpenAI(model=llm_name, temperature=0,max_tokens=50)
        return llm
    
    if (llm_name.startswith("anthropic")):
        os.environ["AWS_PROFILE"] = 'gili-bedrock'
        import boto3
        from langchain.llms.bedrock import Bedrock
        BEDROCK_CLIENT = boto3.client("bedrock-runtime", 'us-east-1')
        return Bedrock(model_id=llm_name, client=BEDROCK_CLIENT, model_kwargs={"temperature": 0})
    assert(False)
    
    
def read_dataset():
    # open CSV file and read first two columns, skip  header row. Use CSV package to do it
    # df = pd.read_csv('./dataset/bank.csv')
    # df = df.drop(columns=[2,3], axis=1)
    # print(df.info())
    import csv
    records = []
    for row in csv.reader(open('./dataset/bank.csv', 'r'), delimiter=',', quotechar='"'):
        # add the first two columns
        records.append({'category': row[0].strip(), 'message': row[1].strip()})
    # remove the first record (header)
    records = records[1:]
    return records

def print_records(records):
    # TODO improve printout https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    for record in records:
        new_record = record.copy()
        new_record['message'] = "".join(reversed(record['message']))
        new_record['category'] = "".join(reversed(record['category']))
        logger.log(logging.DEBUG, new_record)
        
        
def get_unique_categories(records):
    categories = set()
    for record in records:
        categories.add(record['category'])
    # sort the categories set for consistency across executions
    return sorted(categories)

        
def create_few_shot_prompt_template(unique_categories, record_per_unique_category, is_chat : bool, human_nickname : str):  
    categories_csv = ", ".join(unique_categories)
    task_str = f"סווג את ההודעות הבאות לבדיוק אחת מן הקטגוריות הללו: {categories_csv}.",
    examples = record_per_unique_category
    
    example_prompt = PromptTemplate(input_variables=["message", "category"], 
                                    template= f"""
{human_nickname}: הודעה: {{message}}

Assistant: קטגוריה: {{category}}"""
    )
    logger.log(logging.INFO, example_prompt.format(**examples[0]))
    
    fewshot_template = FewShotPromptTemplate(
        examples=examples, 
        example_prompt=example_prompt, 
        prefix=f"{human_nickname}: {task_str}",
        suffix= f"""
{human_nickname}: הודעה: {{message}}

Assistant: קטגוריה: """, 
        input_variables=["message"]
    )
    
    if not is_chat:
        return fewshot_template
    else:
        human_message_prompt = HumanMessagePromptTemplate(prompt=fewshot_template)
        chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
        return chat_prompt_template


def log_to_file_as_json(output_file, the_object):        
    with open(output_file, 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(the_object, indent=4, ensure_ascii=False))
    

def get_results_all_file(output_folder, llm_name):
    return f'{output_folder}/results-all-{llm_name}.json'


def split_train_test(records):
    categories_seen = set()
    train = []
    test = []
    for record in records:
        category = record['category']
        if category not in categories_seen:
            categories_seen.add(category)
            train.append(record)
        else:
            test.append(record)
    return train, test

output_folder = None

def get_output_folder(output_path):
    global output_folder
    if output_folder is not None:
        return output_folder
    
    import os
    from datetime import datetime
    output_folder = f'{output_path}/{datetime.now().strftime("%Y%m%dT%H%M%S")}'
    logger.log(logging.INFO, f"output_folder={output_folder}")
    os.makedirs(output_folder, exist_ok=True) 
    return output_folder

def call_llm(llm_name, prompt_template, prompt_str, message):
    llm = get_llm(llm_name)
    while True:
        try: 
            if (llm_name.startswith("gpt")):
                chain = LLMChain(llm=llm, prompt=prompt_template)
                prediction = chain.run(message)
            else:
                prediction = llm(prompt_str)
            return prediction
        # print any other error
        except Exception as e:
            if 'ThrottlingException' in str(e) or 'overloaded' in str(e) or 'Rate limit' in str(e): # could come in various class types
                import time
                sleep_seconds = 10
                logger.log(logging.INFO, f"throttling or tmp server issue. Retrying in {sleep_seconds}s.")
                time.sleep(sleep_seconds)
                continue
            else:
                raise e
    
def generate_model_eval_input_file(records, prompt_template, output_folder : str, llm_name : str):
    import json
    with open(f'{output_folder}/model_eval_prompt-{llm_name}.jsonl', 'a', encoding='utf-8') as outfile:
        for record in records:
            prompt_str = prompt_template.format(message=record['message']).strip()    
            line = {}
            line['prompt'] = prompt_str
            line['referenceResponse'] = record['category']
            line["category"] = record['category']
            outfile.write(json.dumps(line) + "\n")
            
                
def predict(records, prompt_template, output_folder : str, llm_name : str):
    incorrect_records = []
    for record in records:
        prompt_str = prompt_template.format(message=record['message']).strip()
        with open(f'{output_folder}/prompt-{llm_name}.txt', 'a') as outfile:
            outfile.write(prompt_str + "\n" + "-=-=-="*4 + "\n")

            try:
                prediction = call_llm(llm_name, prompt_template, prompt_str, record['message'])
            except Exception as e:
                logger.log(logging.ERROR, f"Error on message: {record['message']}")
                logger.log(logging.ERROR, f"Error: {e}")
                continue # move on to next message
            # break by newline TODO: find how to add stop sequence.
            prediction = prediction.split("\n")[0].strip()
            prediction = prediction.replace("'", "׳").replace("`", "׳") # need for Claude v2 גרש
            record['prediction'] = prediction
            if (prediction == record['category']):
                record['correct'] = True
            else:
                record['correct'] = False
                incorrect_records.append(record)
            logger.log(logging.INFO, f"{record['correct']} Prediction->{''.join(reversed(prediction))} Actual->{''.join(reversed(record['category']))}")
            
            log_to_file_as_json(get_results_all_file(output_folder, llm_name), records)
            log_to_file_as_json(f'{output_folder}/results-incorrect-{llm_name}.json', incorrect_records)
                    

def is_anthropic(llm_name):
    return llm_name.startswith("anthropic")


def evaluate(llm_name, output_path):  
    print(f"Evaluating using LLM: {llm_name}")
    records = read_dataset()
    print_records(records)
    
    unique_categories = get_unique_categories(records)
    train_set, test_set = split_train_test(records)
   
    is_chat = llm_name.startswith("gpt")
    human_nickname = "Human" if is_anthropic(llm_name) else "user"
    prompt_template = create_few_shot_prompt_template(unique_categories,train_set, is_chat, human_nickname)

    prompt_str = prompt_template.format(message='place message to classify here').strip()
    logger.log(logging.INFO, f"prompt_str={prompt_str}")

    output_folder = get_output_folder(output_path)
    predict(test_set, prompt_template, output_folder, llm_name)
    #generate_model_eval_input_file(test_set, prompt_template, output_folder, llm_name)
    return output_folder
        

def report(llm_name : str, reports_parent_folder : str = None, report_folder : str= None):
    if report_folder is None:
        # get the most recent folder in output_folder
        import glob
        folders = glob.glob(f'{reports_parent_folder}/*')
        most_recent_folder = max(folders, key=os.path.getctime)
        logger.log(logging.INFO, f"Found latest results folder to be: {most_recent_folder}")
        report_folder = most_recent_folder
    results_filename = get_results_all_file(report_folder, llm_name)
    
    logger.log(logging.INFO, f"Loading results file: {results_filename}")
    with open(results_filename, 'r', encoding='utf8') as file:
        results = json.load(file)
        logger.log(logging.DEBUG, results)
        
    # convert dict to DataFrame
    df = pd.DataFrame.from_dict(results)
    logger.log(logging.DEBUG, df)
    
    # calculate accuracy
    true_positives = df.correct.sum()
    total = len(df)
    accuracy = true_positives / total * 100
    logger.log(logging.INFO, f"accuracy: {round(accuracy, 2)}% ({true_positives} / {total})")   
    
    # calculate precision and recall per category
    stats = pd.DataFrame(columns=("category",  "precision",  "recall"))
    for category in get_unique_categories(results):
        logger.log(logging.DEBUG, f'Category: {category[::-1]}')
        true_positives = df[(df.prediction == category) & (df.correct == True)].shape[0] 
        false_positives = df[(df.prediction == category) & (df.correct == False)].shape[0] 
        precision = true_positives / (true_positives + false_positives) * 100
        logger.log(logging.DEBUG, f"  Precision: {round(accuracy, 2)}% ({true_positives} / {(true_positives + false_positives)})")
        
        # calculate recall
        false_negatives = df[(df.category == category) & (df.correct == False)].shape[0] 
        recall = true_positives / (true_positives+false_negatives) * 100
        logger.log(logging.DEBUG, f"  recall: {round(recall, 2)}% ({true_positives} / {(true_positives+false_negatives)})")
        # dd a new row to stats 
        stats = pd.concat([stats, 
                           pd.DataFrame({"category": category, "precision": round(precision, 2), "recall": round(recall, 2)}, index=[0])
                           ], ignore_index=True)
    # print final per category stats]
    logger.log(logging.INFO, "\n" + str(stats))
    
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    # add argument action which can be either x or y
    parser.add_argument(
        "--action", type=str, default="evaluate", choices=('evaluate', 'report') 
    )
    parser.add_argument(
        "--llm-names", type=str, default='anthropic.claude-instant-v1', 
        choices=('j2-grande', 'anthropic.claude-v2', 'anthropic.claude-instant-v1', 'anthropic.claude-v1', 'text-davinci-003', 'gpt-4','gpt-3.5-turbo')
    )
    parser.add_argument(
        "-o", "--output-path", type=str, help="Where to save all the output", default="./benchmark_output"
    )
    args = parser.parse_args()
    return args   

    
def main():
    args = parse_args()
    
    for llm_name in args.llm_names.split(","):
        llm_name = llm_name.strip()
        if args.action == "evaluate":
            report_folder = evaluate(llm_name=llm_name,
                        output_path=args.output_path
            )
            report(llm_name=llm_name,
                report_folder=output_folder
            )
        elif args.action == "report":
            report(llm_name=llm_name,
                reports_parent_folder=args.output_path
            )
        else:
            assert False, f"unknown action: {args.action}"
        
        
if __name__ == "__main__":
    main()
