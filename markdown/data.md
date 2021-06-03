# Bioinformatics Modeling
## Data Collection and Organization


## Synthetic Gene Sequence Data Builder

![png](images/assay_data.png)

We use a synthetic data generator to create a set of base sequences.
A  motif  pattern is inserted into a subset of these sequences.
Sequences with the motif are labled as having a binding property.
Sequences without the motif are labeled as not having a binding probperty.

Assay data is uploaded to ICOS. 

The file is cataloged and added to the project as connected data.


## Enable project_lib interface to faciliate data asset access.

```python
# Use when running on Watson Studio
from project_lib import Project
project = Project.access()
storage_credentials = project.get_storage_metadata()
```


```python
import sys
sys.path.append('.')
```


```python
import string 
import random
import pandas as pd
import numpy as np
```

### Function Definitions

#### Build gene sequence feature and label lists


```python
def build_assay_data(numseq=200, seqlen=50,motif='CGACCGAACTCC'):
    len_motif=len(motif)
    limit=seqlen-len_motif
    binary_choice= range(2)

    seqf=[]
    labf=[]
    #locf=[]
    
    num_insertions=0

    for i in range(numseq):

        seqx = ''.join(random.choices('ATGC' , k = seqlen)) 
        bind=random.choice(binary_choice)
        if bind==1:
            index=random.randrange(1,limit)
            seqx = seqx[:index] + motif + seqx[index + len_motif:]
            num_insertions+=1
        else: 
            bind=0
            index=0
        #print(seqx,bind)
        seqf.append(seqx)
        labf.append(bind)
        #locf.append(index)
        
    print("numseq:",numseq,"seqlen:",seqlen,"number insertions:",num_insertions)
    
    return seqf,labf

```

#### Save assay data


```python
def save_assay_data(sequences,sequence_fn, labels, label_fn):
    print ("Saving file: {}".format(sequence_fn))
    fd=open(sequence_fn,'w')
    for element in sequences:
         fd.write(element)
         fd.write('\n')
    fd.close() 

    print ("Saving file: {}".format(label_fn))
    fd=open(label_fn,'w')
    for element in labels:
         fd.write(str(element))
         fd.write('\n')
    fd.close()
```

#### Build assay csv file


```python
def build_assay_csv_file(sequences,labels, csv='assay_data.csv'):
    
    assay_data=[]
 
    assay_data.append(['sequence','target'])
    for item in zip(sequences,labels):
        assay_data.append([item[0],item[1]])
     
    df=None
    
    df=pd.DataFrame(assay_data, index=np.arange(1, len(assay_data)+1), 
                 columns=['Sequence','BindProperty']) 

    df.to_csv(csv, index = False,header=False)
        
    print("Saving file '{}'".format(csv))
    print("assay_data record count:",len(assay_data))
    
    return df

```

#### Assay data file reader


```python
def read_assay_data_file(csvfile,splitfile=True):
    
    sequences=[]
    labels=[]
    f = open(csvfile, "r")
    for x in f:
      item=x.strip('\n').split(',')  
      sequences.append(item[0])
      labels.append(int(item[1]))
    
    if splitfile:
        prefix=csvfile.split('.')[0]
        
        sequence_fn=prefix+'.seq'
        label_fn=prefix+'.lbl'
        
        print ("Saving file: {}".format(sequence_fn))
        fd=open(sequence_fn,'w')
        for element in sequences:
             fd.write(element)
             fd.write('\n')
        fd.close() 
        
        print ("Saving file: {}".format(label_fn))
        fd=open(label_fn,'w')
        for element in labels:
             fd.write(str(element))
             fd.write('\n')
        fd.close()
        
    return sequences, labels
```

## Build files...


```python
numseq=200
seqlen=100
motif='CGACCGAACTCC'
#choice_range=50

sequences, labels = build_assay_data(numseq=numseq,seqlen=seqlen,motif=motif)
save_assay_data(sequences,'assay_data_full.seq',labels,'assay_data_full.lbl')
build_assay_csv_file(sequences,labels,csv=r'assay_data_full.csv')
```

    numseq: 200 seqlen: 100 number insertions: 95
    Saving file: assay_data_full.seq
    Saving file: assay_data_full.lbl
    Saving file 'assay_data_full.csv'
    assay_data record count: 201





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sequence</th>
      <th>BindProperty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>sequence</td>
      <td>target</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TGGCCGTTCAAGGGCATTAGAACAACGTGCGTTCTTTCTACCCTGC...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GGGCACCCATGGATAGGGCCGCTGTGTCGCGGTAATCGACCGAACT...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AGTTAAACCACACCCTACTCGATGCTTCAGTGGCTTTGGAATTGCG...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GCAAAATTTCTCTGTCAGTATCAGAGTACGTGATCTCTCTACCCAT...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>197</th>
      <td>ACCGTATGGAGGAACCCAGATCGAACGTACGTACGACCGAACTCCG...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>198</th>
      <td>CGGACGACCGAACTCCTCCATCCTAGAACCCATATGAGACCAATAT...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>199</th>
      <td>ATCATACCAGACAACGACCGAACTCCTGTCAAAACCGTGGCCTATA...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>200</th>
      <td>CCATTTCCGTGAGTACTAGCTCCGTAGTGGTACGTCAGATGAGCAC...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>201</th>
      <td>CTAATCCCCTTGGGGTAGGTTCTAGCAATCGTGGGAACCCGGCGCG...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>201 rows × 2 columns</p>
</div>



```python
!ls -al
```
<details>

    total 2224
    
    -rw-r--r--   1 fjgreco  staff    6493 Aug 13 23:19 ICOS.py
    -rw-r--r--   1 fjgreco  staff       1 Aug 13 23:19 README.md
    drwxr-xr-x   8 fjgreco  staff     256 Aug 25 00:12 [34mRESULTS[m[m
    -rw-r--r--   1 fjgreco  staff    3246 Aug 13 23:19 SALIENCE.py
    -rw-r--r--   1 fjgreco  staff    2938 Aug 13 23:19 SALIENCE2.py
    -rw-r--r--   1 fjgreco  staff    6097 Aug 13 23:19 WKC.py
    -rw-r--r--   1 fjgreco  staff   20616 Aug 25 09:48 assay_data_full.csv
    -rw-r--r--   1 fjgreco  staff     400 Aug 25 09:48 assay_data_full.lbl
    -rw-r--r--   1 fjgreco  staff   20200 Aug 25 09:48 assay_data_full.seq
    -rw-r--r--   1 fjgreco  staff     435 Aug 24 22:44 assay_folder.json
    drwxr-xr-x   2 fjgreco  staff      64 Aug 25 08:26 [34mcheckpoints[m[m
    -rw-r--r--   1 fjgreco  staff    4851 Aug 13 23:19 icos_utilities.py
    -rw-r--r--   1 fjgreco  staff   25135 Aug 25 09:47 mac-bioinformatics-assayV3.ipynb
    -rw-r--r--   1 fjgreco  staff   45579 Aug 25 09:34 mac-neural_network_build-brnn.ipynb
    -rw-r--r--   1 fjgreco  staff   21339 Aug 25 09:35 mac-neural_network_build.ipynb
    -rw-r--r--   1 fjgreco  staff     704 Aug 24 22:26 manifest.json
    -rw-r--r--   1 fjgreco  staff    7620 Aug 25 09:26 neural_network8.py
    -rw-r--r--   1 fjgreco  staff    7620 Aug 25 09:25 new_neural_network.py
    -rw-r--r--   1 fjgreco  staff    4023 Aug 13 23:19 tf_model_v6.zip
    -rw-r--r--   1 fjgreco  staff  101137 Aug 13 23:19 wml-mac-bioinformatics-deployment.ipynb
    -rw-r--r--   1 fjgreco  staff   35799 Aug 13 23:19 wml-mac-bioinformatics-e2e.ipynb
    -rw-r--r--   1 fjgreco  staff  718036 Aug 24 22:41 wml-mac-bioinformatics-salience.ipynb
    -rw-r--r--   1 fjgreco  staff   50760 Aug 24 23:50 wml-mac-bioinformatics-train.ipynb
</details>

### Build  separate sequence and label file from full  assay data file (Optional)

```python
s,l=read_assay_data_file('assay_data_full.csv',splitfile=True)
```
## Review list of created assets


```python
!pwd
!ls -al assay*
```

    /Users/fjgreco/GitHub/E2EAI-private/sourcecode
    -rw-r--r--  1 fjgreco  staff  20616 Aug 25 09:48 assay_data_full.csv
    -rw-r--r--  1 fjgreco  staff    400 Aug 25 09:48 assay_data_full.lbl
    -rw-r--r--  1 fjgreco  staff  20200 Aug 25 09:48 assay_data_full.seq
    -rw-r--r--  1 fjgreco  staff    435 Aug 24 22:44 assay_folder.json
    
    assay:
    total 792
    -rw-r--r--   1 fjgreco  staff  200616 Aug 25 08:00 assay_data_full.csv
    -rw-r--r--   1 fjgreco  staff     400 Aug 25 07:51 assay_data_full.lbl
    -rw-r--r--   1 fjgreco  staff  200200 Aug 25 07:51 assay_data_full.seq



```python
with open("e2eai_credentials.json") as json_file:
    credentials = json.load(json_file)

NIRVANA=credentials["NIRVANA"] # /v2
NIRVANA_AUTH=credentials["NIRVANA_AUTH"]
nirvana_apikey=credentials['nirvana_apikey'] # APIKey obtained from CP4D dashboard
nirvana_credentials=credentials['nirvana_credentials']

cp4d_template=credentials['cp4d_template']

IBMCLOUD=credentials['IBMCLOUD_DATA']  #/v2/catalogs/

IBMCLOUD_AUTH=credentials["IBMCLOUD_AUTH"]
        
ibmcloud_apikey=credentials['ibmcloud_apikey']

ibmcloud_template=credentials['ibmcloud_template']

wml_credentials=credentials['wml_credentials_e2eai']

icos_credentials=credentials['icos_credentials_e2eai']

apikey=icos_credentials['apikey'] #ICOS-E2EAI-credentials 

ri=icos_credentials['resource_instance_id']

#display(cp4d_template)
#display(ibmcloud_template)
#print(ri)
#print(apikey,nirvana_apikey,nirvana_credentials)
```

## Save assets in ICOS


```python
import ICOS as ICOS
import WKC as WKC
import json
```


```python
bucket='assay-060221'

asset_name="assay-060221"

connection_path="/assay-060221/assay"
```


```python
icos=ICOS.ICOS(icos_credentials=icos_credentials)
```


```python
icos.create_bucket(bucket)
```




    s3.Bucket(name='assay-060221')

```python
icos.upload_file(bucket,'assay_data_full.csv','assay-060321/assay/assay_data_full.csv')
```

    File Uploaded



```python
icos.upload_file(bucket,'assay_data_full.lbl','assay-060321/assay_data_full.lbl')
```

    File Uploaded



```python
icos.upload_file(bucket,'assay_data_full.seq','assay-060321/assay_data_full.seq')
```

    File Uploaded






## Catalog assets in WKC


```python
wkc=WKC.WKC(ibmcloud_template,catalog_name='Catalog-010721')
```

    Platfom type is cloud.Calling gen_BearerToken_ibmcloud
     Desired catalog_name is set to 'Catalog-010721'.
    guid is set to 'c28ff3ef-3fa0-4c6c-a7f8-c2be92976a75'.



```python
metadata={
  "metadata": {
    "name": asset_name,
    "description": "RNA sequences for transcription binding site motif analysis",
    "tags": ["RNA", "Assay"],
    "asset_type": "folder_asset",
    "origin_country": "us",
    "rov": {
      "mode": 0
    }
  },
  "entity": {
    "folder_asset": {
      "connection_id": wkc.get_asset_id('Connection to project COS','connection'),
      "connection_path": connection_path
    }
  }
}
```

    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/urllib3/connectionpool.py:847: InsecureRequestWarning: Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
      InsecureRequestWarning)



```python
json_metadata = json.dumps(metadata, indent = 2)
```


```python
with open('assay_folder.json', 'w') as fd:
    fd.write(json_metadata)
```
#### Alternative Approach - Inline editing

```python
%%writefile assay_folder.json
{
  "metadata": {
    "name": "Assay-071420",
    "description": "RNA sequences for transcription binding site motif analysis",
    "tags": ["RNA", "Assay"],
    "asset_type": "folder_asset",
    "origin_country": "us",
    "rov": {
      "mode": 0
    }
  },
  "entity": {
    "folder_asset": {
      "connection_id": "3dc06240-b46d-452c-aebf-5821df8a11d4",
      "connection_path": "/assay-071420/assay"
    }
  }
}
```

```python
wkc.add_asset('assay_folder.json')
```

    {
      "metadata": {
        "rov": {
          "mode": 0,
          "collaborator_ids": {}
        },
        "usage": {
          "last_updated_at": "2021-06-03T03:45:37Z",
          "last_updater_id": "IBMid-100000UUG2",
          "last_update_time": 1622691937927,
          "last_accessed_at": "2021-06-03T03:45:37Z",
          "last_access_time": 1622691937927,
          "last_accessor_id": "IBMid-100000UUG2",
          "access_count": 0
        },
        "name": "assay-060221",
        "description": "RNA sequences for transcription binding site motif analysis",
        "tags": [
          "RNA",
          "Assay"
        ],
        "asset_type": "folder_asset",
        "origin_country": "us",
        "resource_key": "assay-060221",
        "rating": 0.0,
        "total_ratings": 0,
        "catalog_id": "c28ff3ef-3fa0-4c6c-a7f8-c2be92976a75",
        "created": 1622691937927,
        "created_at": "2021-06-03T03:45:37Z",
        "owner_id": "IBMid-100000UUG2",
        "size": 0,
        "version": 2.0,
        "asset_state": "available",
        "asset_attributes": [
          "folder_asset"
        ],
        "asset_id": "463135ab-a4ce-44ee-bce7-eec4b8e063ef",
        "asset_category": "USER"
      },
      "entity": {
        "folder_asset": {
          "connection_id": "Not Found",
          "connection_path": "/assay-060221/assay"
        }
      },
      "href": "https://api.dataplatform.cloud.ibm.com/v2/assets/463135ab-a4ce-44ee-bce7-eec4b8e063ef?catalog_id=c28ff3ef-3fa0-4c6c-a7f8-c2be92976a75",
      "asset_id": "463135ab-a4ce-44ee-bce7-eec4b8e063ef"
    }



```python

```


[![return](../buttons/return.png)](../README.md#Data)
[![return](../buttons/next.png)](./neural_network.md)

