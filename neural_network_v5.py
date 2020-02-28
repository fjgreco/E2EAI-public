#%load /project_data/data_asset/new_neural_network4.py
import argparse
#import input_data
import os
import sys
import tensorflow 
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers 
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser()

    # environment variable when name starts with $
    parser.add_argument('--data_dir', type=str, default='$DATA_DIR',help='Directory with data')
    parser.add_argument('--result_dir', type=str, default='$RESULT_DIR',help='Directory with results')
    parser.add_argument('--sequences_file', type=str,default='sequences.txt',help='File name for sequences')
    parser.add_argument('--labels_file', type=str,default='labels.txt',help='File name for labels')
    parser.add_argument('--model_name', type=str,default='bioinformatics_model',help='neural model name')

    FLAGS, unparsed = parser.parse_known_args()

    print (FLAGS.result_dir)

    if (FLAGS.result_dir[0] == '$'):
        RESULT_DIR = os.environ[FLAGS.result_dir[1:]]
    else:
        RESULT_DIR = FLAGS.result_dir
        os.environ['RESULT_DIR']=FLAGS.result_dir

    #model_path = os.path.join(RESULT_DIR, 'model')
    #print(model_path)

    if (FLAGS.data_dir[0] == '$'):
        DATA_DIR = os.environ[FLAGS.data_dir[1:]]
    else:
        DATA_DIR = FLAGS.data_dir
        os.environ['DATA_DIR']=FLAGS.data_dir
        
    output_model_folder = os.environ["RESULT_DIR"]

    print("output model folder: ",output_model_folder)
    
    model_name=FLAGS.model_name
    
    h5_filename  = model_name+".h5"
    print("h5_filename: ",h5_filename)
    
    tar_filename = model_name+".tgz"
    print("tar_filename: ",tar_filename)
    
    model_weights = model_name + "_weights.h5"
    print("model_weights: ", model_weights)
    
    serialized_model = model_name + ".json"
    print("serialized_model: ", serialized_model)
    
    loss_graph_pdf = model_name + "_loss.pdf"
    loss_graph_png = model_name + "_loss.png"
    
    accuracy_graph_pdf = model_name + "_accuracy.pdf"
    accuracy_graph_png = model_name + "_accuracy.png"
    
    confusion_matrix_png = model_name + "_confusion_matrix.png"
    
    scoring_log = model_name + "_scoring.txt"
    

    # Add data dir to file path
    sequences_file = os.path.join(DATA_DIR, FLAGS.sequences_file)
    
    labels_file = os.path.join(DATA_DIR, FLAGS.labels_file)
    
    #
    # One-hot encode feature data
    #
    
    with open(sequences_file,'r') as file: 
        raw_sequences=file.read()

    sequences=raw_sequences.split('\n')

    sequences = list(filter(None, sequences))  # Removes empty sequences.

    integer_encoder = LabelEncoder() 

    one_hot_encoder = OneHotEncoder(categories='auto')  
    
    input_features = []

    for sequence in sequences:
        integer_encoded = integer_encoder.fit_transform(list(sequence))
        integer_encoded = np.array(integer_encoded).reshape(-1, 1)
        one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
        input_features.append(one_hot_encoded.toarray())


    np.set_printoptions(threshold=40)
    input_features = np.stack(input_features)
    print("Sequence 1\n-----------------------")
    print('DNA Sequence #1:\n',sequences[0][:10],'...',sequences[0][-10:])
    print('One hot encoding of Sequence #1:\n',input_features[0].T)

    #
    # One-hot encode labels
    #
    with open(labels_file,'r') as file: 
        raw_labels=file.read()

    labels=raw_labels.split('\n')

    labels = list(filter(None, labels))  # This removes empty sequences.

    one_hot_encoder = OneHotEncoder(categories='auto')
    labels = np.array(labels).reshape(-1, 1)
    input_labels = one_hot_encoder.fit_transform(labels).toarray()

    print('Labels:\n',labels.T)
    print('One-hot encoded labels:\n',input_labels.T)

    train_features, test_features, train_labels, test_labels = train_test_split(
        input_features, input_labels, test_size=0.25, random_state=42)
        
            
    #
    # Define the neural network model
    #    
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=12, 
                 input_shape=(train_features.shape[1], 4)))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', 
        metrics=['binary_accuracy'])
    model.summary()
    
    #
    # Train the model
    #
    
    history = model.fit(train_features, train_labels, 
            epochs=50, verbose=0, validation_split=0.25)
    
    #
    # Produce accuracy and loss graphs
    #
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.savefig('bioinformatics_model_loss.png')
    plt.savefig('bioinformatics_model_loss.pdf')
    #plt.show()
    
    
    plt.figure()
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig('bioinformatics_model_accuracy.png')
    plt.savefig('bioinformatics_model_accuracy.pdf')
    #plt.show()
    
    
    #
    # Save model
    #
    
    model.save( h5_filename ) 
    cmdstring1 = 'cp ' + h5_filename + ' '+  output_model_folder
    os.system(cmdstring1)

    cmdstring2 = 'tar -zcvf ' + tar_filename + ' ' + h5_filename
    os.system(cmdstring2)
    
    cmdstring22 = 'cp ' + tar_filename + ' '+  output_model_folder
    os.system(cmdstring22)
    
    
    #
    # serialize model to JSON
    #
    model_json = model.to_json()
    with open(serialized_model, "w") as json_file:
        json_file.write(model_json)     
 
    cmdstring3 = 'cp ' + serialized_model + ' '+  output_model_folder
    os.system(cmdstring3)

    #
    # Save model weights
    #
    model.save_weights(model_weights)
    cmdstring4 = 'cp ' + model_weights + ' '+  output_model_folder
    os.system(cmdstring4)
    
    #
    # Save loss and accuracy graphs
    #
    cmdstring5 = 'cp ' + loss_graph_pdf + ' '+  output_model_folder
    os.system(cmdstring5)

    cmdstring6 = 'cp ' + loss_graph_png + ' '+  output_model_folder
    os.system(cmdstring6)

    cmdstring7 = 'cp ' + accuracy_graph_pdf + ' '+  output_model_folder
    os.system(cmdstring7)

    cmdstring8 = 'cp ' + accuracy_graph_png + ' '+  output_model_folder
    os.system(cmdstring8)
    
    ## Produce a confusion matrix
    from sklearn.metrics import confusion_matrix
    import itertools

    predicted_labels = model.predict(np.stack(test_features))
    cm = confusion_matrix(np.argmax(test_labels, axis=1), 
                          np.argmax(predicted_labels, axis=1))
    #print('Confusion matrix:\n',cm)

    cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    plt.colorbar()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.xticks([0, 1]); plt.yticks([0, 1])
    plt.grid('off')
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > 0.5 else 'black')
        
    plt.savefig('bioinformatics_model_confusion_matrix.png')
    
    cmdstring9 = 'cp ' + confusion_matrix_png + ' '+  output_model_folder
    os.system(cmdstring9)
    
    from IPython.utils.io import Tee
    from contextlib import closing
    
    with closing(Tee(scoring_log, "w", channel="stdout")) as outputstream:
        model.summary()
        scores = model.evaluate(test_features, test_labels, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        # raise Exception('The file "outputfile.log" is closed anyway.')
        
    cmdstring10 = 'cp ' + scoring_log + ' '+  output_model_folder
    os.system(cmdstring10)
    
if __name__ == '__main__':
    
    main()
