import argparse
from relation_extract import *
from preprocessing.IP_data_generation import *
from SelfORE.evaluate import eval_class
from knowledge_graph import get_kg, find_entity

def run(file_path, model_path=None):
    print("Start preparing data..")
    sent = create_ip_file(file_path)
    print("Finished preparing data")
    print("Start inference...")
    max_idx = eval_class("data/input.json", model_path=model_path)
    #print(max_idx)
    print("Finished inference")

    ent = find_entity(sent)
    # empty list to read sentences from a file
    text = []
    # open file and read the content in a list
    with open(r'data/input.json', 'r') as fp:
        data = json.load(fp)
        text = [x["text"] for x in data]
    	# for line in fp:
    	#     x = line[:-1]
        #     text.append(x)
    relation = []
    for d in text:
    	current_text = get_text_between_entities(d)
    	relation.append(get_relation(current_text))

    dt = get_relations(max_idx,text)
    #print(dt)
    get_kg(relation, ent[0], ent[1], "root", dt)
    get_kg(max_idx, ent[0], ent[1], "n_gram", dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the analysis of a PDF document.")

    parser.add_argument("--file_path", type=str, default="data/file.pdf", help="PDF file path")
    parser.add_argument("--model_path", type=str, default=None, help="model file path")

    args = parser.parse_args()

    config = {key: val for key, val in vars(args).items() if val is not None}
    run(**config)
