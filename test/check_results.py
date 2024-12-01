import os
import json

path = "logs.json"

with open(path, "r") as f:
    data = json.load(f)

print("len of data: ", len(data))

avg_accuracy = 0
avg_completeness = 0
avg_structural = 0

for i in range(len(data)):
    avg_accuracy += float(data[i]["accuracy_score"][0])
    avg_completeness += float(data[i]["completeness_score"][0])
    avg_structural += float(data[i]["structural_score"][0])

avg_accuracy /= len(data)
avg_completeness /= len(data)
avg_structural /= len(data)

print("avg accuracy: ", avg_accuracy)
print("avg completeness: ", avg_completeness)
print("avg structural: ", avg_structural)
# Assign weightage to each score
weight_accuracy = 0.5
weight_completeness = 0.3
weight_structural = 0.2

total_acc = (avg_accuracy * weight_accuracy + 
             avg_completeness * weight_completeness + 
             avg_structural * weight_structural)

print(total_acc)

# import matplotlib.pyplot as plt
# import numpy as np

# # Extract accuracy, completeness, and structural scores
# accuracy_scores = [float(entry["accuracy_score"][0]) for entry in data]
# completeness_scores = [float(entry["completeness_score"][0]) for entry in data]
# structural_scores = [float(entry["structural_score"][0]) for entry in data]

# # Plot accuracy scores in a separate window
# plt.figure(figsize=(17, 4))
# plt.bar(np.arange(len(data)), accuracy_scores, color='blue')
# plt.title('Accuracy Scores')
# plt.xlabel('Data Index')
# plt.ylabel('Score')
# plt.grid(True)
# plt.show()

# # Plot completeness scores in a separate window
# plt.figure(figsize=(17, 4))
# plt.bar(np.arange(len(data)), completeness_scores, color='green')
# plt.title('Completeness Scores')
# plt.xlabel('Data Index')
# plt.ylabel('Score')
# plt.grid(True)
# plt.show()

# # Plot structural scores in a separate window
# plt.figure(figsize=(17, 4))
# plt.bar(np.arange(len(data)), structural_scores, color='red')
# plt.title('Structural Scores')
# plt.xlabel('Data Index')
# plt.ylabel('Score')
# plt.grid(True)
# plt.show()

# # Create a line plot to visualize all scores together in a separate window
# plt.figure(figsize=(17, 8))
# plt.plot(np.arange(len(data)), accuracy_scores, label='Accuracy Score', color='blue', marker='o')
# plt.plot(np.arange(len(data)), completeness_scores, label='Completeness Score', color='green', marker='o')
# plt.plot(np.arange(len(data)), structural_scores, label='Structural Score', color='red', marker='o')

# # Add labels and title
# plt.xlabel('Data Index')
# plt.ylabel('Scores')
# plt.title('Evaluation Scores Over Data Entries')
# plt.legend()

# # Show grid
# plt.grid(True)

# # Display the plot
# plt.tight_layout()
# plt.show()


print("Generated data:")
print("Average input tokens: ", sum([entry['gen_input_tokens'] for entry in data]) / len(data))
print("Average output tokens: ", sum([entry['gen_output_tokens'] for entry in data]) / len(data))
print("Average time taken: ", sum([entry['gen_time_spent'] for entry in data]) / len(data))
print("Average cost: ", sum([entry['gen_cost'] for entry in data]) / len(data))
print("Total cost: ", sum([entry['gen_cost'] for entry in data]))


print("Evaluated data:")
print("Average input tokens: ", sum([entry['eval_input_tokens'] for entry in data]) / len(data))
print("Average output tokens: ", sum([entry['eval_output_tokens'] for entry in data]) / len(data))
print("Average time taken: ", sum([entry['eval_time_spent'] for entry in data]) / len(data))
print("Total cost: ", sum([entry['eval_cost'] for entry in data]))

print("Total time taken: ", sum([entry['gen_time_spent'] for entry in data]) + sum([entry['eval_time_spent'] for entry in data]))
print("Total cost: ", sum([entry['gen_cost'] for entry in data]) + sum([entry['eval_cost'] for entry in data]))