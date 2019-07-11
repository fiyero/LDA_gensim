# Build a LDA model for classification with Gensim
## https://medium.com/@patrickhk/build-a-lda-model-for-classification-with-gensim-80ca6343c4b9

### Objective
Our goal is to build a LDA model to classify news into different category/(topic).<br/>
### Dataset
We will use the abcnews-date-text.csv provided by udaicty. It contains over 1 million entries of news headline over 15 years. The dataset have two columns, the publish date and headline.<br/>

### Data preprocessing to extract data
We use pandas to read the csv and select the first 300000 entries as our dataset instead of using all the 1 million entries. Our model will likely be more accurate if using all entries.<br/>


![p1](https://miro.medium.com/max/444/1*zOvgE6_OptV6d5F9wgWBuA.png)<br/>
Then we carry out usual data cleansing, including removing stop words, stemming, lemmatization, turning into lower case..etc after tokenization.<br/>
So our processed corpus will be in this form, each document is a list of token, instead of a raw text string.<br/>

![p2](https://miro.medium.com/max/511/1*tQfdFssvNou2yebtY5cEGg.png)<br/>

### Get the Bag of word dict

To build LDA model with Gensim, we need to feed corpus in form of Bag of word dict or tf-idf dict.
```dictionary = gensim.corpora.Dictionary(processed_docs)```
We filter our dict to remove key : value pairs with less than 15 occurrence or more than 10% of total number of sample<br/>
```dictionary.filter_extremes(no_below=15, no_above=0.1)```
### Convert into bag of word format for doc2bow

![p3](https://miro.medium.com/max/585/1*JoZHtg2m8VydhkSSgUtv7g.png)<br/>
### (Optional) Get the td-idf corpus form

![p4](https://miro.medium.com/max/700/1*mbL4l25V21iGmvAFZy6Qfw.png)<br/>
### Run LDA model using BOW with gensim
```lda_model = gensim.models.LdaMulticore(bow_corpus,num_topics=10,id2word = dictionary,passes = 2,workers=2)```
Here I choose num_topics=10, we can write a function to determine the optimal number of the paramter, which will be discussed later.     ### Interpret the result
                                  
![p5](https://miro.medium.com/max/1000/1*-wBRq5MA0AL0_XUTKA7ODA.png)<br/>
I only show part of the result in here. Since we set num_topic=10, the LDA model will classify our data into 10 difference topics.
Each topic is a combination of keywords and each keyword contributes a certain weight to the topic. For example 0.04* ”warn” mean token “warn” contribute to the topic with weight =0.04<br/>
The result will only tell you the integer label of the topic, we have to infer the identity by ourselves. For example topic 1 have keywords “gov, plan, council, water, fund…etc” so it makes sense to guess topic 1 is related to politics.<br/>
We can also run the LDA model with our td-idf corpus, can refer to my github at the end.<br/>

### Evaluation of LDA model
One common way is to calculate the topic coherence with c_v<br/>

![p6](https://miro.medium.com/max/700/1*RUewNf8M2nf3fxom4kU9ZQ.png)<br/>
### Get the optimal num_topic for LDA model
write a function to calculate the coherence score with varying num_topics parameter then plot graph with matplotlib<br/>
```def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
return model_list, coherence_values```

![p7](https://miro.medium.com/max/434/1*eQNTOt8XVbRCzplTVwjv3Q.png)<br/>
### Try the LDA model with unseen data
Lets say our testing news have headline “My name is Patrick”, pass the headline to the SAME data processing step and convert it into BOW input then feed into the model<br/>
![p7](https://miro.medium.com/max/700/1*0goRi9M9BtXTUGe10aFR7A.png)<br/>

It seems our LDA model classify our “My name is Patrick” news into the topic of “politics”.<br/>
Why? It is possible many political news headline contain People’ name or title as keyword.<br/>
Anyways this is just a toy LDA model, we can see some keywords in the LDA result are actually fragment instead of complete vocab. For example we can see “charg” and “chang”, which should be “charge” and “change”. This is due to imperfect data processing step.<br/>


-------------------------------------------------------------------------------------------------------------------------------------
### More about me
[[:pencil:My Medium]](https://medium.com/@patrickhk)<br/>
[[:house_with_garden:My Website]](https://www.fiyeroleung.com/)<br/>
[[:space_invader:	My Github]](https://github.com/fiyero)<br/>
