import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plotting(file_name):
    with open(file_name, 'r') as f:
        results = json.load(f)
    val_losses = results['val_losses']
    train_losses = results['train_losses']
    val_loss = results['val_loss']
    val_acc = results['val_acc']
    test_loss = results['test_loss']
    test_acc = results['test_acc']
    
    # Plotting the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('./plots/problem1.png')
    plt.show()
    
def plot_difference(y, pred):
    plt.figure(figsize=(20, 6))
    y_flat = y.flatten()
    pred_flat = pred.flatten()
    plt.plot(y_flat, label='True', marker='o', alpha=0.7)
    plt.plot(pred_flat, label='Predicted', marker='x', alpha=0.7)
    plt.title('Temperature Forecast vs. Actual')
    plt.xlabel('Time step')
    plt.ylabel('Temperature (C)')
    plt.legend()
    plt.grid(True)
    plt.savefig('./plots/p2_difference.png')
    plt.show()

if __name__ == "__main__":
    plotting('./results/problem1.json')