![png](images/CommonProjectFramework.png)

## Post Deployment Use

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


[![return](../buttons/return.png)](../README.md#Use)
