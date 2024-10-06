import matplotlib.pyplot as plt
import json
import sys

def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def create_charts(data):
    iterations = []
    losses = []
    accuracies = []
    for d in data:
        if 'iter' in d:
            iterations.append(d["iter"])
        if 'loss' in d:
            losses.append(d["loss"])
        if 'acc' in d:
            accuracies.append(d["acc"])

    print(len(iterations))
    print(len(losses))
    plt.figure(figsize=(12, 5))

    # Loss chart
    plt.subplot(1, 2, 1)
    plt.plot(iterations, losses, marker='o')
    plt.title('Loss over iterations')
    plt.xlabel('iter')
    plt.ylabel('Loss')

    # Accuracy chart
    plt.subplot(1, 2, 2)
    plt.plot(iterations, accuracies, marker='o')
    plt.title('Accuracy over iterations')
    plt.xlabel('iter')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = read_data_from_file(file_path)
    create_charts(data)
