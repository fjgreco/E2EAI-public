## Deployment

## DEPLOYMENT

### Save re-compiled model and package for deployment
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




[![return](../buttons/return.png)](../README.md#Deployment)
