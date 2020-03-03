

![png](images/WMLProcess.png)


# Bioinformatics Modeling Using Watson Machine Learning


```python
from project_lib import Project
project = Project.access()
storage_credentials = project.get_storage_metadata()
```

### Preliminary:  Instantiate a Watson Studio project , WML, ICOS (done outside of this notebook)
<details>

```python
!pip uninstall  wget -y
!pip install wget

#!pip uninstall watson-machine-learning-client-V4 -y 
#!pip install watson-machine-learning-client-V4

!pip uninstall watson-machine-learning-client -y
!pip install watson-machine-learning-client==1.0.378
```

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
```
</details>

### WML Client Version 1.0378 setup for demo purposes

<details>
```python
import sys, os, imp
```


```python
print(sys.path)
```

    ['', '/cc-home/_global_/python-3.6', '/cc-home/_global_/python-3', '/user-home/_global_/python-3.6', '/user-home/_global_/python-3', '/opt/conda/envs/Python-3.6/lib/python36.zip', '/opt/conda/envs/Python-3.6/lib/python3.6', '/opt/conda/envs/Python-3.6/lib/python3.6/lib-dynload', '/home/wsuser/.local/lib/python3.6/site-packages', '/opt/conda/envs/Python-3.6/lib/python3.6/site-packages', '/opt/conda/envs/Python-3.6/lib/python3.6/site-packages/IPython/extensions', '/home/wsuser/.ipython']



```python
sys.path.reverse()
from watson_machine_learning_client import WatsonMachineLearningAPIClient
sys.path.reverse()
print("Done")
```

    Done

</details>

## Connect to Watson Machine Learning

Note: These credentials are unhidden for demo purposes.


```python
wml_credentials = {
  "apikey": "AvtF6zGC6YD2FHWiB9viJTFtzTbBZecyPzAYzh0Q6ISy",
  "iam_apikey_description": "Auto-generated for key fc7e3c6e-98a2-4f10-81c4-edcb67ab5ba4",
  "iam_apikey_name": "Service credentials-1",
  "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Writer",
  "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/c095a1d4738948bf43a84feb693c0d52::serviceid:ServiceId-989b8f81-f13f-4ca0-a2a3-3f71a1b51db7",
  "instance_id": "615a5483-2072-4c9d-8d76-3fed527c6b59",
  "url": "https://us-south.ml.cloud.ibm.com"
}

client = WatsonMachineLearningAPIClient(wml_credentials)

print('client.version:', client.version)
```

    client.version: 1.0.378


## Connect to IBM Cloud Object Storage and define search script

<details>

```python
icos_credentials={
  "apikey": "uSYOgQRcMVikimnEwccgQcNW7xwp-ckXyB-8H-qhb2u-",
  "cos_hmac_keys": {
    "access_key_id": "7b602b4cefa5483e8ee7d731e3275c52",
    "secret_access_key": "3ac90cff6372194eb258b07f761bea92b7b1740923d3ad9e"
  },
  "endpoints": "https://control.cloud-object-storage.cloud.ibm.com/v2/endpoints",
  "iam_apikey_description": "Auto-generated for key 7b602b4c-efa5-483e-8ee7-d731e3275c52",
  "iam_apikey_name": "HIMSS2020-Credential",
  "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Writer",
  "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/c095a1d4738948bf43a84feb693c0d52::serviceid:ServiceId-24edfd07-f9ab-46f0-be92-d8203b252328",
  "resource_instance_id": "crn:v1:bluemix:public:cloud-object-storage:global:a/c095a1d4738948bf43a84feb693c0d52:b342a808-6323-46bd-844b-e01bc0671f9b::"
}

import ibm_boto3
from ibm_botocore.client import Config, ClientError
from datetime import datetime
import time


# Constants for IBM COS values
COS_ENDPOINT = "https://s3.us.cloud-object-storage.appdomain.cloud" # Current list avaiable at https://control.cloud-object-storage.cloud.ibm.com/v2/endpoints
COS_API_KEY_ID = icos_credentials["apikey"] # eg "W00YiRnLW4a3fTjMB-odB-2ySfTrFBIQQWanc--P3byk"
COS_AUTH_ENDPOINT = 'https://iam.bluemix.net/oidc/token'
COS_RESOURCE_CRN = icos_credentials["resource_instance_id"]

training_bucket = 'hmss2020-training'
results_bucket  = 'hmss2020-results'

# Create resource
cos = ibm_boto3.resource("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_RESOURCE_CRN,
    ibm_auth_endpoint=COS_AUTH_ENDPOINT,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)
cos_client = ibm_boto3.client("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_RESOURCE_CRN,
    ibm_auth_endpoint=COS_AUTH_ENDPOINT,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)

def get_download_list(bucket_name,model_location,cos=cos):
    
    print("Retrieving relevant bucket contents from: {} Model_location: {}\n".format(bucket_name,model_location)) 
    download_list=[]
    try:
        files = cos.Bucket(bucket_name).objects.all()
        for file in files: 
            item=file.key.split('/')
            if 'bioinformatics' in item[1] and model_location in item[0]:
                print(file.key)
                download_list.append(file.key)
                fn=file.key.split('/')[1]
                cos_client.download_file('hmss2020-results',Key=file.key,Filename=fn)
                     
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to retrieve bucket contents: {0}".format(e))
    return download_list  
```
</details>

##  Package the model program and supporting artifacts into a zip file and pass to Watson Machine Learning for training...

### Approach 1:  Obtain zip file from project asset storage

<details>

```python
!cp /project_data/data_asset/tf_model_v5.zip .
```
</details>

### Approach 2: Obtain zip file from git hub
<details>

import os
import wget
filename = 'tf_model_v5.zip'

#url= 'https://github.com/fjgreco/himss2020/blob/master/tf_model_v5.zip?raw=true'

url= 'https://github.com/fjgreco/Bioinformatics/blob/master/tf_model_v5.zip?raw=true'


# NOTE: If you are re-running this code block again, having changed the model or adding your own custom model
# be careful to ensure that your new model is the one which is truly downloaded.

if not os.path.isfile( filename ):
    print("File {} not found, Downloading from {}".format(filename,url))
    wget.download(url)
</details>

### Approach 3: Build the zip file inline
<details>

%%bash
(if [ ! -d "tf_model_v5" ] 
 then mkdir tf_model_v5 
 fi
 cp /project_data/data_asset/neural_network_v5.py tf_model_v5
 zip -r tf_model_v5.zip tf_model_v5 )
## Define a  WML training run

</details>

### We use a timestamp to distinguish different training runs and results

```python
import datetime

print('Timestamp: {:%Y%m%d%H%M%S}'.format(datetime.datetime.now()))
ts='{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
print(ts)
```

    Timestamp: 20200302233424
    20200302233424


### The following code identifies the following WML metadata:
<ul>
<li>The runtime environment in which our code will run i.e., Tensorflow V 1.14 and Python 3.6
<li>The python command that WML will use to execute the program. 
<li>The location of the zip file that contains the machine learning model program, which will be passed to the WML server.
</ul>


```python
#V3
metadata = {
client.repository.DefinitionMetaNames.NAME              : "zen-bioinformatics-training_v3nn5_"+ts,
client.repository.DefinitionMetaNames.FRAMEWORK_NAME    : "tensorflow",
client.repository.DefinitionMetaNames.FRAMEWORK_VERSION : "1.14",
client.repository.DefinitionMetaNames.RUNTIME_NAME      : "python",
client.repository.DefinitionMetaNames.RUNTIME_VERSION   : "3.6",
client.repository.DefinitionMetaNames.EXECUTION_COMMAND : "python3 tf_model_v5/neural_network_v5.py --sequencesFile ${DATA_DIR}/sequences.txt --labelsFile ${DATA_DIR}/labels.txt" 
}
definition_details = client.repository.store_definition( "tf_model_v5.zip", meta_props=metadata )
definition_uid     = client.repository.get_definition_uid( definition_details )
print( "definition_uid: ", definition_uid )
```

    definition_uid:  f205f565-21bb-444a-a4a1-abb03f847d24



```python

```

### The following code identifies the following WML metadata:
<ul>
<li>The IBM Cloud Object Storage bucket where our input training data resides 
<li>The IBM Cloud Object Storage bucket where our results will end up. 
<li>The hardware platform on which we want to run, in this case K80.
</ul>


```python
icos_metadata = {
client.training.ConfigurationMetaNames.NAME         : "zen-bioinformatics-training-run_v3nn5_"+ts,
client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCE : {
   "connection" : { 
      "endpoint_url"      : "https://s3.us.cloud-object-storage.appdomain.cloud",
      "access_key_id"     : "7d6ca104f1fb4ff88bd8ad9ec7bdd090",
      "secret_access_key" : "f89db50acd602219b4e5aa443cca7b2ca8f949ad0ca927ed"
      },
   "source" : { 
      "bucket" : "hmss2020-training",
      },
      "type" : "s3"
   },
client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {
   "connection" : {
      "endpoint_url"      : "https://s3.us.cloud-object-storage.appdomain.cloud",
      "access_key_id"     : "7d6ca104f1fb4ff88bd8ad9ec7bdd090",
      "secret_access_key" : "f89db50acd602219b4e5aa443cca7b2ca8f949ad0ca927ed"
      },
      "target" : {
         "bucket" : "hmss2020-results",
      },
      "type" : "s3"
   }
}
print('Done')
```

    Done


## Submit the training definition to the WML server for training and monitor run progress


```python
run_details = client.training.run(definition_uid, meta_props=icos_metadata)

run_uid     = client.training.get_run_uid(run_details)
print("run_uid: ", run_uid)
```

    run_uid:  model-k34afm4o



```python
from time import sleep

cts=client.training.get_details(run_uid)['entity']['status']['state']

while cts not in ['completed', 'failed', 'canceled', 'error']:
    print(cts)
    sleep(10)
    cts=client.training.get_status(run_uid)['state']
    
print( cts)
```
<details>
    pending
    pending
    pending
    pending
    pending
    pending
    pending
    running
    running
    running
    running
    running
    running
    running
    running
    running
    running
    running
    running
    completed
</details>        


### Check  training run processing details:


```python
client.training.get_details(run_uid)
```


<details>

    {'entity': {'model_definition': {'definition_href': 'https://us-south.ml.cloud.ibm.com/v3/ml_assets/training_definitions/f205f565-21bb-444a-a4a1-abb03f847d24',
       'execution': {'command': 'python3 tf_model_v5/neural_network_v5.py --sequencesFile ${DATA_DIR}/sequences.txt --labelsFile ${DATA_DIR}/labels.txt',
        'compute_configuration': {'name': 'k80'}},
       'framework': {'name': 'tensorflow', 'version': '1.14'},
       'name': 'zen-bioinformatics-training-run_v3nn5_20200302233424'},
      'status': {'current_at': '2020-03-02T23:40:47.919Z',
       'finished_at': '2020-03-02T23:39:39.769Z',
       'message': 'training-WyAzx1lWg: ',
       'metrics': [],
       'running_at': '2020-03-02T23:37:44.450Z',
       'state': 'completed',
       'submitted_at': '2020-03-02T23:37:33.627Z'},
      'training_data_reference': {'connection': {'access_key_id': '7d6ca104f1fb4ff88bd8ad9ec7bdd090',
        'endpoint_url': 'https://s3.us.cloud-object-storage.appdomain.cloud',
        'secret_access_key': 'f89db50acd602219b4e5aa443cca7b2ca8f949ad0ca927ed'},
       'location': {'bucket': 'hmss2020-training'},
       'type': 's3'},
      'training_results_reference': {'connection': {'access_key_id': '7d6ca104f1fb4ff88bd8ad9ec7bdd090',
        'endpoint_url': 'https://s3.us.cloud-object-storage.appdomain.cloud',
        'secret_access_key': 'f89db50acd602219b4e5aa443cca7b2ca8f949ad0ca927ed'},
       'location': {'bucket': 'hmss2020-results',
        'model_location': 'training-WyAzx1lWg'},
       'type': 's3'}},
     'metadata': {'created_at': '2020-03-02T23:37:30.982Z',
      'guid': 'model-k34afm4o',
      'modified_at': '2020-03-02T23:37:30.982Z',
      'url': '/v3/models/model-k34afm4o'}}

</details>

## Review  the training results generated by the machine model program


```python
ctd=client.training.get_details(run_uid)
model_location= ctd['entity']['training_results_reference']['location']['model_location'] 
```

### Download training results from shared object storage to CP4D/WS working storage. 


```python
  dl2=get_download_list(results_bucket,model_location)
```

    Retrieving relevant bucket contents from: hmss2020-results Model_location: training-WyAzx1lWg
    
    training-WyAzx1lWg/bioinformatics_model.h5
    training-WyAzx1lWg/bioinformatics_model.json
    training-WyAzx1lWg/bioinformatics_model.tgz
    training-WyAzx1lWg/bioinformatics_model_accuracy.pdf
    training-WyAzx1lWg/bioinformatics_model_accuracy.png
    training-WyAzx1lWg/bioinformatics_model_confusion_matrix.png
    training-WyAzx1lWg/bioinformatics_model_loss.pdf
    training-WyAzx1lWg/bioinformatics_model_loss.png
    training-WyAzx1lWg/bioinformatics_model_scoring.txt
    training-WyAzx1lWg/bioinformatics_model_weights.h5



```python
image = mpimg.imread("bioinformatics_model_accuracy.png")
plt.imshow(image)
plt.show()
```


![png](output_36_0.png)



```python
image = mpimg.imread("bioinformatics_model_loss.png")
plt.imshow(image)
plt.show()
```


![png](output_37_0.png)



```python
image = mpimg.imread("bioinformatics_model_confusion_matrix.png")
plt.imshow(image)
plt.show()
```


![png](output_38_0.png)



```python
!cat bioinformatics_model_scoring.txt
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv1d (Conv1D)              (None, 39, 32)            1568      
    _________________________________________________________________
    max_pooling1d (MaxPooling1D) (None, 9, 32)             0         
    _________________________________________________________________
    flatten (Flatten)            (None, 288)               0         
    _________________________________________________________________
    dense (Dense)                (None, 16)                4624      
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 34        
    =================================================================
    Total params: 6,226
    Trainable params: 6,226
    Non-trainable params: 0
    _________________________________________________________________
    binary_accuracy: 97.40%


### Store results in project_asset storage

```bash
%%bash -s "$ts"
(if [ ! -d "/project_data/data_asset/tf_model_v5_"$1 ] 
 then mkdir /project_data/data_asset/tf_model_v5_$1 
 fi
 cp bioinformatics* /project_data/data_asset/tf_model_v5_$1/.)
```

## SOME TIME LATER...

### Using  the stored model definition and trained weights, we can recompile the model and resume work.


```python
client.repository.list()
```

    ------------------------------------  ------------------------------------------------  ------------------------  ---------------  -----------------
    GUID                                  NAME                                              CREATED                   FRAMEWORK        TYPE
    f205f565-21bb-444a-a4a1-abb03f847d24  zen-bioinformatics-training_v3nn5_20200302233424  2020-03-02T23:37:12.867Z  tensorflow       definition
    7fe962f4-3cc5-4d5d-8e72-8654971ae9e9  zen-bioinformatics-training-definition_v3nn5      2020-03-01T02:00:45.065Z  tensorflow       definition
    7c43ab5d-b52a-490a-9750-a49d92a8ebb7  ws-bioinformatics-training-definition-v5          2020-02-26T00:26:13.427Z  tensorflow       definition
    18479c6b-1688-41da-996b-9cff9d3ecbb9  DL3                                               2020-02-26T04:01:57.495Z  -                experiment
    91af05a1-b7ed-4825-b39c-2e5bb3218c09  zen_bioinformatics_model_v3nn5_r                  2020-03-01T02:07:45.559Z  tensorflow-1.14  model
    7c26209f-7cf0-4103-90ab-6c4c518e032d  DL3-Model-V1.1                                    2020-02-26T04:28:48.160Z  tensorflow-1.14  model
    dbceee12-7835-4aa8-90ec-3fdfa485654a  DL3-model-Version1                                2020-02-26T04:05:29.891Z  tensorflow-1.14  model
    2917fae8-ebdd-435e-af94-281c5ea9015a  ws-bioinformatics_model-v5                        2020-02-26T00:34:08.045Z  tensorflow-1.14  model
    c042c672-5111-4322-a7eb-cf0c4d3177d3  zen_bioinformatics_deployment_v3nn5(r)            2020-03-01T02:07:53.584Z  tensorflow-1.14  online deployment
    eab67438-b5d0-42f4-ad0f-b46cc887f936  ws_bioinformatic_deployment_v5                    2020-02-26T00:34:13.106Z  tensorflow-1.14  online deployment
    ------------------------------------  ------------------------------------------------  ------------------------  ---------------  -----------------



```python
!ls /project_data/data_asset
```

    DATADIR
    DATADIR.zip
    Rnorvegicus.fasta
    __pycache__
    apikey.json
    default_commit_hash
    icoskeys.txt
    labels_txt_54t5ig9cxtblv9s5lvrpc3c1g
    male.hg19.fasta
    male.hg19.fasta copy.fai
    male.hg19.fasta.fai
    neural_network_v5.py
    new_neural_network2.py
    new_neural_network4.py
    object_subfolder
    rbr.out
    results.csv
    sra_repository
    swab_Illumina_fastq_7zts604faspjjdibpea5dk5pm
    tf-model3.zip
    tf-model4
    tf-model4.zip
    tf_model_v5
    tf_model_v5.zip
    tf_model_v5_20200302233424



```python
print ('Effective timestamp: ',ts)
```

    Effective timestamp:  20200302233424


### Retrieve the model artifacts to working storage...


```python
!cp /project_data/data_asset/tf_model_v5_{ts}/* .
```


```python
mh5="bioinformatics_model.h5"
mjson="bioinformatics_model.json"
mtgz="bioinformatics_model.tgz"
mweights="bioinformatics_model_weights.h5"
mconfusion="bioinformatics_model_confusion_matrix.png"

!ls -al bioinformatics*
```

    -rw-r-----. 1 wsuser watsonstudio 108728 Mar  3 01:24 bioinformatics_model.h5
    -rw-r-----. 1 wsuser watsonstudio   1988 Mar  3 01:24 bioinformatics_model.json
    -rw-r-----. 1 wsuser watsonstudio  70339 Mar  3 01:24 bioinformatics_model.tgz
    -rw-r-----. 1 wsuser watsonstudio  12782 Mar  3 01:24 bioinformatics_model_accuracy.pdf
    -rw-r-----. 1 wsuser watsonstudio  25694 Mar  3 01:24 bioinformatics_model_accuracy.png
    -rw-r-----. 1 wsuser watsonstudio  24261 Mar  3 01:24 bioinformatics_model_confusion_matrix.png
    -rw-r-----. 1 wsuser watsonstudio  11819 Mar  3 01:24 bioinformatics_model_loss.pdf
    -rw-r-----. 1 wsuser watsonstudio  25737 Mar  3 01:24 bioinformatics_model_loss.png
    -rw-r-----. 1 wsuser watsonstudio   1036 Mar  3 01:24 bioinformatics_model_scoring.txt
    -rw-r-----. 1 wsuser watsonstudio    143 Mar  3 01:24 bioinformatics_model_v3nn5_20200302233424.url
    -rw-r-----. 1 wsuser watsonstudio  42168 Mar  3 01:24 bioinformatics_model_weights.h5


### Re-compile neural model
This step does not involve training data.


```python
from tensorflow.keras.models import model_from_json
# load json and create model
with open(mjson, 'r') as json_file:
    loaded_model_json = json_file.read()
    
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(mweights)
print("Loaded model from project assets")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

    Loaded model from project assets


## Select data for analysis 

<details>
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np

def ohe_Xy(sequences,labels):
    sequences_file=sequences

    labels_file=labels
    
    with open(sequences_file,'r') as file: 
        raw_sequences=file.read()

    sequences=raw_sequences.split('\n')

    sequences = list(filter(None, sequences))  # This removes empty sequences.


    # The LabelEncoder encodes a sequence of bases as a sequence of integers.
    integer_encoder = LabelEncoder()  
    # The OneHotEncoder converts an array of integers to a sparse matrix where 
    # each row corresponds to one possible value of each feature.
    one_hot_encoder = OneHotEncoder(categories='auto')   
    input_features = []

    for sequence in sequences:
        integer_encoded = integer_encoder.fit_transform(list(sequence))
        integer_encoded = np.array(integer_encoded).reshape(-1, 1)
        one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
        input_features.append(one_hot_encoded.toarray())

    np.set_printoptions(threshold=40)
    input_features = np.stack(input_features)
    #print("One hot encoding of features and labels\n-----------------------")
    #print('\nDNA Sequence #1:\n',sequences[0][:10],'...',sequences[0][-10:])
    #print('\nOne hot encoding of Sequence #1:\n',input_features[0].T)


    with open(labels_file,'r') as file: 
            raw_labels=file.read()

    labels=raw_labels.split('\n')

    labels = list(filter(None, labels))  # This removes empty sequences.

    one_hot_encoder = OneHotEncoder(categories='auto')
    labels = np.array(labels).reshape(-1, 1)
    input_labels = one_hot_encoder.fit_transform(labels).toarray()

    #print('\nLabels:\n',labels.T)
    #print('\nOne-hot encoded labels:\n',input_labels.T)
    
    return input_features, input_labels, sequences

    
def split_input_data(input_features,sequences):
    print("\nSplitting input data into train and test segments...")
    train_features, test_features, train_labels, test_labels = train_test_split(
        input_features, input_labels, test_size=0.25, random_state=42)
    return test_features, test_labels, train_features, train_labels
```


```python
input_features, input_labels, sequences = ohe_Xy('/project_data/data_asset/DATADIR/sequences.txt','/project_data/data_asset/DATADIR/labels.txt')

X,Y,_,_= split_input_data(input_features,input_labels)

print('Done')
```

    
    Splitting input data into train and test segments...
    Done
</details>

### Score model using this data


```python
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
```

    acc: 97.40%


### Produce a confusion matrix


```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

predicted_labels = loaded_model.predict(np.stack(X))
cm = confusion_matrix(np.argmax(Y, axis=1), 
                      np.argmax(predicted_labels, axis=1))
print('Confusion matrix:\n',cm)

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
```

    Confusion matrix:
     [[252   7]
     [  6 235]]



![png](output_59_1.png)


### Produce predictions and salience charts


```python
import tensorflow.keras.backend as K

def compute_salient_bases(model, x):
  input_tensors = [loaded_model.input]
  gradients = model.optimizer.get_gradients(model.output[0][1], model.input)
  compute_gradients = K.function(inputs = input_tensors, outputs = gradients)
  
  x_value = np.expand_dims(x, axis=0)
  gradients = compute_gradients([x_value])[0][0]
  sal = np.clip(np.sum(np.multiply(gradients,x), axis=1),a_min=0, a_max=None)
  return sal
```


```python
sequences2=[
'CCGAGGGCTATGGTTTGGAAGTTAGAACCCTGGGGCTTCTCGCGGACACC',
'GAGTTTATATGGCGCGAGCCTAGTGGTTTTTGTACTTGTTTGTCGCGTCG',
'GATCAGTAGGGAAACAAACAGAGGGCCCAGCCACATCTAGCAGGTAGCCT',
'GTCCACGACCGAACTCCCACCTTGACCGCAGAGGTACCACCAGAGCCCTG',
'GGCGACCGAACTCCAACTAGAACCTGCATAACTGGCCTGGGAGATATGGT',
'AGACATTGTCAGAACTTAGTGTGCGCCGCACTGAGCGACCGAACTCCGAC']
```

### One hot encode sequence data


```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# The LabelEncoder encodes a sequence of bases as a sequence of integers.
integer_encoder = LabelEncoder()  
# The OneHotEncoder converts an array of integers to a sparse matrix where 
# each row corresponds to one possible value of each feature.
one_hot_encoder = OneHotEncoder(categories='auto')  

input_features2 = []
for sequence2 in sequences2:
    #print(sequence2)
    integer_encoded2 = integer_encoder.fit_transform(list(sequence2))
    integer_encoded2 = np.array(integer_encoded2).reshape(-1, 1)
    one_hot_encoded2 = one_hot_encoder.fit_transform(integer_encoded2)
    input_features2.append(one_hot_encoded2.toarray())

np.set_printoptions(threshold=40)
input_features2 = np.stack(input_features2)
#print("Example nucleotide sequence\n-----------------------")
#print('DNA Sequence2 #1:\n',sequences2[0][:10],'...',sequences2[0][-10:])
#print('One hot encoding of Sequence2 #1:\n',input_features2[0].T)
print("Done")
```

    Done


### Make predictions


```python
prediction2=loaded_model.predict(input_features2)
```


```python
def saliency2(model,sequence_index,motif_start):
    sal = compute_salient_bases(model, input_features2[sequence_index])

    plt.figure(figsize=[16,5])
    barlist = plt.bar(np.arange(len(sal)), sal)
    #[barlist[i].set_color('C3') for i in range(motif_start,motif_start+12)]  # Change the coloring here if you change the sequence index.
    plt.xlabel('Bases')
    plt.ylabel('Magnitude of saliency values')
    plt.xticks(np.arange(len(sal)), list(sequences2[sequence_index]));
    plt.title('Saliency map for bases in one of the positive sequences'
              ' (green indicates the actual bases in motif)');
    plt.show()
```


```python
for i in range(len(sequences2)):
    print('\nKnown motif: CGACCGAACTCC')
    print("Sequence:",sequences2[i])
    saliency2(loaded_model,i,37)
    print("Nobind probability: {} Bind probability: {}".format(prediction2[i][0],prediction2[i][1]))
    print("\n-----------------------")
    
```

    
    Known motif: CGACCGAACTCC
    Sequence: CCGAGGGCTATGGTTTGGAAGTTAGAACCCTGGGGCTTCTCGCGGACACC



![png](output_68_1.png)


    Nobind probability: 0.9999997615814209 Bind probability: 2.558667233643064e-07
    
    -----------------------
    
    Known motif: CGACCGAACTCC
    Sequence: GAGTTTATATGGCGCGAGCCTAGTGGTTTTTGTACTTGTTTGTCGCGTCG



![png](output_68_3.png)


    Nobind probability: 1.0 Bind probability: 3.5872671499959097e-09
    
    -----------------------
    
    Known motif: CGACCGAACTCC
    Sequence: GATCAGTAGGGAAACAAACAGAGGGCCCAGCCACATCTAGCAGGTAGCCT



![png](output_68_5.png)


    Nobind probability: 0.9999997615814209 Bind probability: 2.0125524713421328e-07
    
    -----------------------
    
    Known motif: CGACCGAACTCC
    Sequence: GTCCACGACCGAACTCCCACCTTGACCGCAGAGGTACCACCAGAGCCCTG



![png](output_68_7.png)


    Nobind probability: 3.4863812743424205e-06 Bind probability: 0.999996542930603
    
    -----------------------
    
    Known motif: CGACCGAACTCC
    Sequence: GGCGACCGAACTCCAACTAGAACCTGCATAACTGGCCTGGGAGATATGGT



![png](output_68_9.png)


    Nobind probability: 0.0008279726607725024 Bind probability: 0.9991720914840698
    
    -----------------------
    
    Known motif: CGACCGAACTCC
    Sequence: AGACATTGTCAGAACTTAGTGTGCGCCGCACTGAGCGACCGAACTCCGAC



![png](output_68_11.png)


    Nobind probability: 1.6434263443443342e-06 Bind probability: 0.9999983310699463
    
    -----------------------


## DEPLOYMENT

### Save re-compliled model and package for deployment
Note: The compression is required by Keras


```python
loaded_model.save('zen_bioinformatics_model_v3nn5_r.h5')

!tar -zcvf zen_bioinformatics_model_v3nn5_r.tgz zen_bioinformatics_model_v3nn5_r.h5
```

    zen_bioinformatics_model_v3nn5_r.h5


### Define the deployment metadata that will be passed to Watson Machine Learning


```python
metadata={
          client.repository.ModelMetaNames.NAME : "zen_bioinformatics_model_v3nn5_r_"+ts,
          client.repository.ModelMetaNames.FRAMEWORK_NAME :"tensorflow",
          client.repository.ModelMetaNames.FRAMEWORK_VERSION : "1.14",
          client.repository.ModelMetaNames.FRAMEWORK_LIBRARIES : [{'name':'keras', 'version': '2.1.6'}]}

model_details = client.repository.store_model( model="zen_bioinformatics_model_v3nn5_r.tgz", meta_props=metadata )

model_uid = client.repository.get_model_uid(model_details)
print("model_uid: ", model_uid)
```

    model_uid:  7bfbfe1b-9092-4cfd-b75f-797c0b2774dc


### Deploy the model


```python
deployment_name  = "zen_bioinformatics_deployment_v3nn5(r)_"+ts
deployment_desc  = "Deployment of the bioinformatics model v4nn55_"+ts
deployment_details = client.deployments.create(model_uid, name="zen_bioinformatics_deployment_v3nn5(r)_" +ts)

```

    
    
    #######################################################################################
    
    Synchronous deployment creation for uid: '7bfbfe1b-9092-4cfd-b75f-797c0b2774dc' started
    
    #######################################################################################
    
    
    INITIALIZING
    DEPLOY_IN_PROGRESS..
    DEPLOY_SUCCESS
    
    
    ------------------------------------------------------------------------------------------------
    Successfully finished deployment creation, deployment_uid='0f0510b9-fb4e-49d2-a19d-64cbe828334a'
    ------------------------------------------------------------------------------------------------
    
    


### Obtain a scoring endpoint for online use of the  trained model


```python
scoring_endpoint = client.deployments.get_scoring_url(deployment_details)
print("scoring_endpoint: ", scoring_endpoint)
```

    scoring_endpoint:  https://us-south.ml.cloud.ibm.com/v3/wml_instances/615a5483-2072-4c9d-8d76-3fed527c6b59/deployments/0f0510b9-fb4e-49d2-a19d-64cbe828334a/online


### Save the endpoint as a project asset


```python
epfn='bioinformatics_model_v3nn5_'+ts+ '.url'
with open(epfn,'w') as fd:
    fd.write(scoring_endpoint)
```


```python
!cp {epfn} /project_data/data_asset/.
```

## List deployments


```python
client.deployments.list()
```

    ------------------------------------  -----------------------------------------------------  ------  --------------  ------------------------  ---------------  -------------
    GUID                                  NAME                                                   TYPE    STATE           CREATED                   FRAMEWORK        ARTIFACT TYPE
    0f0510b9-fb4e-49d2-a19d-64cbe828334a  zen_bioinformatics_deployment_v3nn5(r)_20200302233424  online  DEPLOY_SUCCESS  2020-03-03T00:09:02.519Z  tensorflow-1.14  model
    c042c672-5111-4322-a7eb-cf0c4d3177d3  zen_bioinformatics_deployment_v3nn5(r)                 online  DEPLOY_SUCCESS  2020-03-01T02:07:53.584Z  tensorflow-1.14  model
    eab67438-b5d0-42f4-ad0f-b46cc887f936  ws_bioinformatic_deployment_v5                         online  DEPLOY_SUCCESS  2020-02-26T00:34:13.106Z  tensorflow-1.14  model
    ------------------------------------  -----------------------------------------------------  ------  --------------  ------------------------  ---------------  -------------


## SOME MORE TIME LATER...

### Retrieve the scoring endpoint from project asset storage


```python
!ls /project_data/data_asset/
```

    DATADIR
    DATADIR.zip
    Rnorvegicus.fasta
    __pycache__
    apikey.json
    default_commit_hash
    icoskeys.txt
    labels_txt_54t5ig9cxtblv9s5lvrpc3c1g
    male.hg19.fasta
    male.hg19.fasta copy.fai
    male.hg19.fasta.fai
    neural_network_v5.py
    new_neural_network2.py
    new_neural_network4.py
    object_subfolder
    results.csv
    sra_repository
    swab_Illumina_fastq_7zts604faspjjdibpea5dk5pm
    tf-model3.zip
    tf-model4
    tf-model4.zip
    tf_model_v5
    tf_model_v5.zip
    tf_model_v5_20200302233424



```python
!cat /project_data/data_asset/tf_model_v5_20200302233424/*url
```

    https://us-south.ml.cloud.ibm.com/v3/wml_instances/615a5483-2072-4c9d-8d76-3fed527c6b59/deployments/0f0510b9-fb4e-49d2-a19d-64cbe828334a/online


```python
scoring_endpoint="https://us-south.ml.cloud.ibm.com/v3/wml_instances/615a5483-2072-4c9d-8d76-3fed527c6b59/deployments/0f0510b9-fb4e-49d2-a19d-64cbe828334a/online"
```

### Set up a one-hot encoded data payload to be evaluated


```python
payload = { "values" : X.tolist() }
```

### Score the data


```python
sc=client.deployments.score(scoring_endpoint, payload)

print(sc['fields'])

for item in sc['values'][0:3]:
    print(item)
```

    ['prediction', 'prediction_classes', 'probability']
    [[1.8571682858237182e-07, 0.9999997615814209], 1, [1.8571682858237182e-07, 0.9999997615814209]]
    [[0.9999996423721313, 3.675713173834083e-07], 0, [0.9999996423721313, 3.675713173834083e-07]]
    [[0.00014102058776188642, 0.9998589754104614], 1, [0.00014102058776188642, 0.9998589754104614]]



```python

```


[![return](../buttons/return.png)](../README.md#WML)
