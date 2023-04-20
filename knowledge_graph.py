import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from SelfORE.evaluate import eval_class
import re

#extract the entities from the text using pattern matching. Search for words between [E] and [/E] tags
def find_entity(text):
    pattern1 = "\[E1\](.*?)\[/E1\]"
    pattern2 = "\[E2\](.*?)\[/E2\]"
    entity_1 = re.findall(pattern1, str(text))
    entity_2 = re.findall(pattern2, str(text))
    return entity_1, entity_2

# Create dataframe of entities and predicted relationships
def get_kg(relations, entity_1, entity_2, type, relation_dict):
        
    kg_df = pd.DataFrame({'entity_1': entity_1, 'entity_2': entity_2, 'edge': relations})
    #remove rows which empty entities
    for i in range(len(kg_df)):
    	if(kg_df['entity_1'][i] == '' or kg_df['entity_2'][i] == ''):
    		kg_df.drop([i], axis=0, inplace=True)

    # Create network from dataframe
    G=nx.from_pandas_edgelist(kg_df, "entity_1", "entity_2", edge_attr="edge", create_using=nx.DiGraph())
    G.remove_edges_from(nx.selfloop_edges(G)) #remove self loop
    
    # Plot the graph
    if(type == "root"):
	    plt.figure(figsize=(15,8))
	    plt.title('Root word KG')
	    pos = nx.spring_layout(G, seed=13, k=0.5)
	    nx.draw(G, with_labels=True, edge_cmap=plt.cm.Blues, pos = pos, font_size = 8, node_shape = "s", node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.2'))
	    edge_labels = nx.get_edge_attributes(G, "edge")
	    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)
	    plt.show()
    elif(type == "n_gram"):
    	plt.figure(figsize=(15,8))
    	plt.title('N-gram KG')
    	i=0.6
    	for key in relation_dict:
    		plt.text(0.8, i, str(key) + ": " + str(relation_dict[key]), style='italic')
    		i = i+0.2
    	pos = nx.spring_layout(G, seed=13, k=0.5)
    	nx.draw(G, with_labels=True, edge_cmap=plt.cm.Blues, pos = pos, font_size = 8, node_shape = "s", node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.2'))
    	edge_labels = nx.get_edge_attributes(G, "edge")
    	nx.draw_networkx_edge_labels(G, pos, edge_labels)
    	plt.show()
