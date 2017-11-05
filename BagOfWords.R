
library(readr)
amazon <- read_csv("amazon.csv")
google <- read_csv("google.csv")


View(amazon)
str(amazon)
amazon_pros <- amazon$pros
amazon_cons <- amazon$cons


str(google)
google_pros <- google$pros
google_cons <- google$cons



#**Text organization**
  #Now that we have selected the exact text sources, we are now ready to clean them up. We'll be using the two functions qdap_clean(), which applies a series of qdap functions to a text vector, and tm_clean(),which applies a series of tm functions to a corpus object.


library(qdap)
qdap_clean <- function(x){
x<- na.omit(x)
x<- replace_abbreviation(x)
x<- replace_contraction(x)
x<- replace_number(x)
x<- replace_ordinal(x)
x<- replace_symbol(x)
x<-tolower(x)
return(x)
}


library(tm)


tm_clean <- function(x){
x<-tm_map(x,removePunctuation)
x<-tm_map(x,stripWhitespace)
x<-tm_map(x,removeWords,c(stopwords("en"),"Amazon","Google","Company"))
return(x)
}



#Applying qdap_clean() to amazon and google

amazon_pros <- qdap_clean(amazon_pros)
amazon_cons <- qdap_clean(amazon_cons)

google_pros <- qdap_clean(google_pros)
google_cons <- qdap_clean(google_cons)


#Next step is to convert this vector containing the text data to a corpus. Corpus is a collection of documents, but it's also important to know that in the tm domain, R recognizes it as a data type.

##We will use the volatile corpus, which is held in  computer's RAM rather than saved to disk, just to be more memory efficient.

# #To make a volatile corpus, R needs to interpret each element in our vector of text, amazon_pros, as a document.
# And the tm package provides what are called Source functions to do just that! We'll use a Source function called VectorSource() because our text data is contained in a vector. The output of this function is called a Source.


amazon_p_corp <- VCorpus(VectorSource(amazon_pros))
amazon_c_corp <- VCorpus(VectorSource(amazon_cons))

google_p_corp <- VCorpus(VectorSource(google_pros))
google_c_corp <- VCorpus(VectorSource(google_cons))


Now using tm_clean to clean data

amazon_pros_corp <- tm_clean(amazon_p_corp)
amazon_cons_corp <- tm_clean(amazon_c_corp)

google_pros_corp <- tm_clean(google_p_corp)
google_cons_corp <- tm_clean(google_c_corp)




**Steps 4 & 5: Feature extraction & analysis**
  
  #Since amzn_pros_corp, amzn_cons_corp, goog_pros_corp and goog_cons_corp have all been preprocessed, so now we can extract the features we want to examine. Since we are using the bag of words approach, we decide to create a bigram TermDocumentMatrix for Amazon's positive reviews corpus, amzn_pros_corp. From this, we can quickly create a wordcloud() to understand what phrases people positively associate with working at Amazon.

#The function below uses RWeka to tokenize two terms.

library(RWeka)
tokenizer <- function(x) 
NGramTokenizer(x, Weka_control(min = 2, max = 2))

#Feature extraction & analysis: amazon_cons

amazon_p_tdm <- TermDocumentMatrix(amazon_pros_corp)
amazon_p_tdm_m <- as.matrix(amazon_p_tdm)
amazon_p_freq <- rowSums(amazon_p_tdm_m)
amazon_p_f.sort <- sort(amazon_p_freq,decreasing = TRUE)

barplot(amazon_p_freq[1:5])



library(wordcloud)
amazon_p_tdm <- TermDocumentMatrix(amazon_pros_corp,control = list(tokenize=tokenizer))
amazon_p_tdm_m <- as.matrix(amazon_p_tdm)
amazon_p_freq <- rowSums(amazon_p_tdm_m)
amazon_p_f.sort <- sort(amazon_p_freq,decreasing = TRUE)
p_df <- data.frame(term=names(amazon_p_f.sort),num=amazon_p_f.sort)

wordcloud(p_df$term,p_df$num,max.words=100,color="red")

#Feature extraction & analysis: amazon_cons

amazon_c_tdm <- TermDocumentMatrix(amazon_cons_corp,control=list(tokenize=tokenizer))
amazon_c_tdm_m <- as.matrix(amazon_c_tdm)
amazon_c_freq <- rowSums(amazon_c_tdm_m)
amazon_c_f.sort <- sort(amazon_c_freq,decreasing = TRUE)
c_df <- data.frame(term=names(amazon_c_f.sort),num=amazon_c_f.sort)

wordcloud(c_df$term,c_df$num,max.words=100,color="red")


#*amazon_cons dendrogram*

#It seems there is a strong indication of long working hours and poor work-life balance in the reviews. As a simple clustering technique, we'll decide to perform a hierarchical cluster and create a dendrogram to see how connected these phrases are.


amazon_c_tdm <- TermDocumentMatrix(amazon_cons_corp,control = list(tokenize=tokenizer))
amazon_c_tdm <- removeSparseTerms(amazon_c_tdm,0.993)

amazon_c_hclust <- hclust(dist(amazon_c_tdm,method="euclidean"),method="complete")

plot(amazon_c_hclust)


#*Word association*
  #Switching back to positive comments, we'll decide to examine top phrases that appeared in the word clouds. We'll now hope to find associated terms using the findAssocs() function from tm package.


amazon_p_tdm <- TermDocumentMatrix(amazon_pros_corp,control=list(tokenize=tokenizer))
amazon_p_m <- as.matrix(amazon_p_tdm)
amazon_p_freq <- rowSums(amazon_p_m)
token_frequency <- sort(amazon_p_freq,decreasing = TRUE)
token_frequency[1:5]

findAssocs(amazon_p_tdm,"fast paced",0.2)


#We decide to create a comparison.cloud() of Google's positive and negative reviews for comparison to Amazon. This will give you a quick understanding of top terms.


all_google_pros <- paste(google$pros,collapse="")
all_google_cons <- paste(google$cons,collapse = "")

all_google <- c(all_google_pros,all_google_cons)
all_google_qdap <- qdap_clean(all_google)
all_google_vs <- VectorSource(all_google_qdap) 
all_google_vc <- VCorpus(all_google_vs)
all_google_clean<- tm_clean(all_google_vc)
all_google_tdm <- TermDocumentMatrix(all_google_clean)
colnames(all_google_tdm) <- c("Google Pros","Google Cons")
all_google_tdm_m <- as.matrix(all_google_tdm)

comparison.cloud(all_google_tdm_m,colors = c("orange","blue"),max.words = 50)


#Amazon's positive reviews appear to mention bigrams such as "good benefits", while its negative reviews focus on bigrams such as  "work-life balance" issues.

#In contrast to, Google's positive reviews mention  "perks", "smart people","great food", and "fun culture", among other things. Google's negative reviews discuss "politics", "getting big", "bureaucracy", and "middle management".

#Now we'll  make a pyramid plot lining up positive reviews for Amazon and Google so you can adequately see the differences between any shared bigrams.


amazon_pro <- paste(amazon$pros,collapse = "")
google_pro <- paste(google$pros,collapse = "")
all_pro <- c(amazon_pro,google_pro)
all_pro_qdap <- qdap_clean(all_pro)
all_pro_vs <- VectorSource(all_pro)
all_pro_vc <- VCorpus(all_pro_vs)
all_pro_corp <- tm_clean(all_pro_vc)

tdm.bigram = TermDocumentMatrix(all_pro_corp,control = list(tokenize =tokenizer))
colnames(tdm.bigram) <- c("Amazon","Google")
tdm.bigram <- as.matrix(tdm.bigram)
common_words<- subset(tdm.bigram,tdm.bigram[,1] > 0 & tdm.bigram[,2] > 0 )
difference <- abs(common_words[, 1] - common_words[,2])
common_words <- cbind(common_words,difference)
common_words <- common_words[order(common_words[,3],decreasing = TRUE),]
top25_df <- data.frame(x=common_words[1:25,1],y=common_words[1:25,2],labels=rownames(common_words[1:25,]))

library(plotrix)
pyramid.plot(top25_df$x,top25_df$y,labels=top25_df$labels,gap=15,top.labels=c("Amazon Pros","Vs","Google Pros"),unit = NULL,main = "Words in common")


#Amazon employees discussed "work-life balance" as a positive. In both organizations, people mentioned "culture" and "smart people", so there are some similar positive aspects between the two companies.

#You now decide to turn your attention to negative reviews and make the same visuals.


amazon_cons <- paste(amazon$cons,collapse = "")
google_cons <- paste(google$cons,collapse = "")
all_cons <- c(amazon_cons,google_cons)
all_cons_qdap <- qdap_clean(all_cons)
all_cons_vs <- VectorSource(all_cons)
all_cons_vc <- VCorpus(all_cons_vs)
all_cons_corp <- tm_clean(all_cons_vc)

tdm.cons_bigram = TermDocumentMatrix(all_cons_corp,control=list(tokenize =tokenizer))

colnames(tdm.cons_bigram) <- c("Amazon","Google")
tdm.cons_bigram <- as.matrix(tdm.cons_bigram)
common_words<- subset(tdm.cons_bigram,tdm.cons_bigram[,1] > 0 & tdm.cons_bigram[,2] > 0 )
difference <- abs(common_words[, 1] - common_words[,2])
common_words <- cbind(common_words,difference)
common_words <- common_words[order(common_words[,3],decreasing = TRUE),]
top25_df <- data.frame(x=common_words[1:25,1],y=common_words[1:25,2],labels=rownames(common_words[1:25,]))

library(plotrix)
pyramid.plot(top25_df$x,top25_df$y,labels=top25_df$labels,gap=10,top.labels=c("Amazon cons","Vs","Google cons"),unit = NULL,main = "Words in common")

# We wll use Commonality cloud to show common between Aamazon and google with Unigram, Bigram and Trigram tokenizer to identify more insights.

#**Unigram**
  
tdm.unigram <- TermDocumentMatrix(all_pro_corp)
colnames(tdm.unigram) <- c("Amazon","Google")
tdm.unigram <- as.matrix(tdm.unigram)

commonality.cloud(tdm.unigram,colors=c("red","yellow"),max.words = 100)


#**Bigram**
  
  
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
tdm.bigram <- TermDocumentMatrix(all_pro_corp,control = list(tokenize=BigramTokenizer))
colnames(tdm.bigram) <- c("Amazon","Google")
tdm.bigram <- as.matrix(tdm.bigram)

commonality.cloud(tdm.bigram,colors=c("red","yellow"),max.words = 100)


#**Trigram**
  
  
TrigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))
tdm.trigram <- TermDocumentMatrix(all_pro_corp,control = list(tokenize=TrigramTokenizer))
colnames(tdm.trigram) <- c("Amazon","Google")
tdm.trigram <- as.matrix(tdm.trigram)

commonality.cloud(tdm.trigram,colors=c("red","yellow"),max.words = 100)



#Plotting amazon and google association
{r warning=FALSE,message=FALSE}

library(ggthemes)
library(ggplot2)

amazon_tdm <- TermDocumentMatrix(amazon_p_corp)
associations <- findAssocs(amazon_tdm,"fast",0.2)
associations_df <- list_vect2df(associations)[,2:3] 

ggplot(associations_df,aes(y=associations_df[,1]))+
  geom_point(aes(x=associations_df[,2]),
             data=associations_df,size=3)+ 
  theme_gdocs()



library(ggthemes)
library(ggplot2)

google_tdm <- TermDocumentMatrix(google_c_corp)
associations <- findAssocs(google_tdm,"fast",0.2)
associations_df <- list_vect2df(associations)[,2:3] 

ggplot(associations_df,aes(y=associations_df[,1]))+
  geom_point(aes(x=associations_df[,2]),
             data=associations_df,size=3)+ 
  theme_gdocs()


#Conclusion**
  Google have a better work-life balance according to current employee reviews.

findAssocs(amazon_p_tdm, "fast paced", 0.2)[[1]][1:15]

#We Identified candidates that view an intense workload as an opportunity to learn fast and give them ample opportunity.