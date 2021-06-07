#%load /project_data/data_asset/new_neural_network8.py
import argparse
#import input_data
import os
import sys
import tensorflow 
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers 
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten,LSTM
from tensorflow.keras.optimizers import Adam


from tensorflow.keras.models import Sequential

def main():
    
    import os
    """
    cmdstring = 'pip install matplotlib'
    os.system(cmdstring)
    import matplotlib.pyplot as plt
    """
    parser = argparse.ArgumentParser()

    # environment variable when name starts with $
    parser.add_argument('--data_dir', type=str, default='$DATA_DIR',help='Directory with data')
    parser.add_argument('--result_dir', type=str, default='$RESULT_DIR',help='Directory with results')
    parser.add_argument('--sequences_file', type=str,default='sequences.txt',help='File name for sequences')
    parser.add_argument('--labels_file', type=str,default='labels.txt',help='File name for labels')
    parser.add_argument('--model_name', type=str,default='bioinformatics_model',help='neural model name')
    parser.add_argument('--lstm',type=bool,default=True,help='Include LSTM')
    parser.add_argument('--epochs',type=int,default=10,help='Number of epochs')
    parser.add_argument('--lr',type=float,default=0.01,help='Learning rate')
    parser.add_argument("--feature_shape",type=int,default=50,help='Feature shape')


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
    
    history_filename  = model_name+"_history.p"
    print("history_filename: ",history_filename)
    
    cm_filename  = model_name+"_cm.p"
    print("cm_filename: ",cm_filename)
    
    h5_filename  = model_name+".h5"
    print("h5_filename: ",h5_filename)
    
    tar_filename = model_name+".tgz"
    print("tar_filename: ",tar_filename)
    
    model_weights = model_name + "_weights.h5"
    print("model_weights: ", model_weights)
    
    serialized_model = model_name + ".json"
    print("serialized_model: ", serialized_model)
   
    
    scoring_log = model_name + "_scoring.txt"
    
    loss_graph_pdf= model_name + "_loss.pdf"
    loss_graph_png = model_name + "_loss.png"
    print("loss_graph:",loss_graph_png)
    
    accuracy_graph_pdf = model_name + "_accuracy.pdf"
    accuracy_graph_png = model_name + "_accuracy.png"
    print("accuracy_graph:",accuracy_graph_png)
    
    
    #
    # Set training hyperparameters
    #
    
    epochs = FLAGS.epochs
    #epochs = 50
    lr     = FLAGS.lr
    #lr=  0.01
    lstm   = FLAGS.lstm
    feature_shape = FLAGS.feature_shape
    
    #
    # Print hyperparameters to stdout
    #
    
    print('\n')
    print("Number of epochs: ", epochs )
    print("Learning Rate:    ", lr)
    print("Include LSTM:     ", lstm )
    print("Feature Shape:    ", feature_shape )
   
    

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
    if lstm == True:
            model.add(LSTM(feature_shape))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    
    opt = Adam(learning_rate=lr)

    model.compile(loss='binary_crossentropy', optimizer='adam', 
        metrics=['binary_accuracy'])
    model.summary()
    
    #
    # Train the model
    #
    
    history = model.fit(train_features, train_labels, 
            epochs=75,  verbose=0, validation_split=0.25)
    
    import pickle
    with open(history_filename, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    
    cmdstring0 = 'cp ' + history_filename + ' '+  output_model_folder
    os.system(cmdstring0)
    
    #
    # Save model to the results storage
    #
    
    model.save( h5_filename ) 
    cmdstring1 = 'cp ' + h5_filename + ' '+  output_model_folder
    os.system(cmdstring1)

    cmdstring2 = 'tar -zcvf ' + tar_filename + ' ' + h5_filename
    os.system(cmdstring2)
    
    cmdstring22 = 'cp ' + tar_filename + ' '+  output_model_folder
    os.system(cmdstring22)
    
    
    #
    # Save the model definition to the results storage
    #
    model_json = model.to_json()
    with open(serialized_model, "w") as json_file:
        json_file.write(model_json)     
 
    cmdstring3 = 'cp ' + serialized_model + ' '+  output_model_folder
    os.system(cmdstring3)

    #
    # Save  trained model weights to the results storage
    #
    model.save_weights(model_weights)
    cmdstring4 = 'cp ' + model_weights + ' '+  output_model_folder
    os.system(cmdstring4)
    
    
    ## Produce and save a confusion matrix
    from sklearn.metrics import confusion_matrix
    #import itertools

    predicted_labels = model.predict(np.stack(test_features))
    cm = confusion_matrix(np.argmax(test_labels, axis=1), 
                          np.argmax(predicted_labels, axis=1))

    cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    
    with open(cm_filename, 'wb') as file_pi:
        pickle.dump(cm, file_pi)
    
    cmdstringX = 'cp ' + cm_filename + ' '+  output_model_folder
    os.system(cmdstringX)
 
    scores = model.evaluate(test_features, test_labels, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
 
if __name__ == '__main__':
    
    main()
