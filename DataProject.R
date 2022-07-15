##############################################################
# data preparation, modeling, evaluation and documenting     #
# the choice of model for deployment                         #
##############################################################

##################
## Preconditions #
##################

# GL1.csv files in R's working directory 
# for this script to work properly

######################
## Install packages  #
######################

install.packages("class")
install.packages("caret")
install.packages("car")
install.packages("lattice")
install.packages("Hmisc")
install.packages("RWeka")
install.packages("rpart")
install.packages("arules")
install.packages("e1071", dep = TRUE, type = "source")


###################
## Load libraries #
###################

library(class)
library(caret)
library(dplyr)
library(car)
library(lattice)
library(Hmisc)
library(RWeka)
library(rpart)
library(arules)
library("e1071", dep = TRUE, type = "source")


##################
## Load the data #
##################

GL1<-read.csv("GL1.csv")

 
GL1$FTR <- as.factor(
  ifelse(GL1$FTR == "H", "Win", "NotWin"))

####################################
## Data Understanding              #
##                                 #
## Visualization and Summarization #
####################################

# str() function - reports on the structure of the data frame
writeLines("\nstr() output for GL1 data")
str(GL1)

# summary() function - displays a summary for each feature
writeLines("\nsummary() output for GL1 data")
print(summary(GL1))

panel.cor <- function(x, y, digits = 2, cex.cor, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  # correlation coefficient
  r <- cor(x, y)
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste("r= ", txt, sep = "")
  text(0.5, 0.6, txt)
  
  # p-value calculation
  p <- cor.test(x, y)$p.value
  txt2 <- format(c(p, 0.123456789), digits = digits)[1]
  txt2 <- paste("p= ", txt2, sep = "")
  if(p<0.01) txt2 <- paste("p= ", "<0.01", sep = "")
  text(0.5, 0.4, txt2)
}

corr = cor(GL1[,4:15])

# Plot the correlations using circles
png("GL1Correlation.png", width=900, height=500)
corrplot(corr, method="circle")
dev.off()

###########################################################
## Data Preparation                                       #
##                                                        #
## Includes feature selection and transformations such as #
## discretization, normalization, standardization...      #
###########################################################

GL1$HSTR <- GL1$HST / GL1$HS

GL1$ASTR <- GL1$AST / GL1$AS

GL1$STRdf <- GL1$HSTR-GL1$ASTR

GL1$Fdf <- GL1$HF-GL1$AF

GL1$Ydf <- GL1$HY-GL1$AY

GL1$Rdf <- GL1$HR-GL1$AR

GL1$Cdf <- GL1$HC-GL1$AC

minHT <- min(GL1$HT)
maxHT <- max(GL1$HT)
GL1$norm.HT <- (GL1$HT - minHT) / (maxHT - minHT)

GL1$HST <- NULL
GL1$AST <- NULL
GL1$FTHG <- NULL
GL1$FTAG <- NULL
GL1$AS <- NULL
GL1$HS <- NULL
GL1$HF <- NULL
GL1$AF <- NULL
GL1$HC <- NULL
GL1$AC <- NULL
GL1$HT <- NULL
GL1$HY <- NULL
GL1$AY <- NULL
GL1$HSTR <- NULL
GL1$ASTR <- NULL
GL1$HR <- NULL
GL1$AR <- NULL
GL1$AwayTeam <- NULL
GL1$HomeTeam <- NULL
GL1$HST. <- NULL
GL1$AST. <- NULL

GL1 <- GL1[c(2, 1, 4,5,6,7)]

# Create normalized features to be used for KNN
normFeature <- function(data) {
  (data - min(data)) / (max(data) - min(data))
}

GL1$HT.norm <- normFeature(GL1$HT)
GL1$STRdf.norm <- normFeature(GL1$STRdf)
GL1$Fdf.norm <- normFeature(GL1$Fdf)
GL1$Cdf.norm <- normFeature(GL1$Cdf)
GL1$Ydf.norm <- normFeature(GL1$Ydf)
GL1$Rdf.norm <- normFeature(GL1$Rdf)

##############################
## Perform feature selection #
##############################

# Use random forest to choose the most important attributes (e.g. decision tree)
# Inclding sort_group which has a 1.0 correspondence with sort_group_name
GL1.rf.scores <- random.forest.importance(FTR ~ ., GL1)

# Display the feature scores
print(GL1.rf.scores)

##Add a numeric target, using the factor values
#GL1$FTR.num <- as.numeric(GL1$FTR)
#GL1$FTR.num <- NULL

# Build decision tree
tree = rpart(FTR ~ ., GL1[,1:7])

# Get predictions
p = predict(tree, GL1[,1:7], type="c")

# Get count of correct predictions
cp = GL1$FTR == p

# get ratio of correct predictions
sum(cp) / nrow(GL1)

## Use FSelector library for feature selection ##

## Define an evaluation function  for the data using a tree   
evaluator.GL1.tree <- function(subset) {
  # Use k-fold cross validation
  k <- 5
  splits <- runif(nrow(GL1))
  results = sapply(1:k, function(i) {
    test.idx <- (splits >= (i - 1) / k) & (splits < i / k)
    train.idx <- !test.idx
    test <- GL1[test.idx, , drop=FALSE]
    train <- GL1[train.idx, , drop=FALSE]
    tree <- rpart(as.simple.formula(subset, "FTR"), train)
    error.rate = sum(test$FTR != predict(tree, test, type="c")) / nrow(test)
    return(1 - error.rate)
  })
  print(subset)
  print(mean(results))
  return(mean(results))
}

## Use forward greedy search on iris data - tree eval 
subset <- forward.search(names(GL1)[2:7], evaluator.GL1.tree)

# Obtain the selected subset of features
ft <- as.simple.formula(subset, "FTR")

# Display the selected subset of features
print(ft)


################################################
## Modeling                                    #
##                                             #
## KNN                                         #
################################################

# Split GL1 data into train and test sets (60% train)
set.seed(0)
trainSet <- createDataPartition(GL1$FTR, p=.6)[[1]]
GL1.train <- GL1[trainSet,]
GL1.test <- GL1[-trainSet,]

GL1.knn.train <- GL1.train[,8:13]
GL1.knn.test <- GL1.test[,8:13]
GL1.knn.traincl <- GL1.train[,1]

###############
## Evaluation #
###############

# Use the knn() function (in the class library)
# Use KNN with 1, 3 and 10 neighbors
# Note there is no "model" other than the training instances
# In this case we simply need the predictions
GL1.pred.knn.1 <- knn(GL1.knn.train, GL1.knn.test, GL1.knn.traincl)
GL1.pred.knn.3 <- knn(GL1.knn.train, GL1.knn.test, GL1.knn.traincl, k=3)
GL1.pred.knn.10 <- knn(GL1.knn.train, GL1.knn.test, GL1.knn.traincl, k=10)

# Calculation of performance for KNN classifiers
GL1.eval.knn.1 <- confusionMatrix(GL1.pred.knn.1, GL1.test$FTR)
GL1.eval.knn.3 <- confusionMatrix(GL1.pred.knn.3, GL1.test$FTR)
GL1.eval.knn.10 <- confusionMatrix(GL1.pred.knn.10, GL1.test$FTR)

# Display the evaluation results for the KNN classifiers
print(GL1.eval.knn.1$table)
print(GL1.eval.knn.3$table)
print(GL1.eval.knn.10$table)

## Plot decision boundaries for k = 1 to 15 
## Only using petal length and petal width

# Create a simplified data frame with only an id,
# normalized petal length, normalized petal width, 
# and 2-classes (positive class is GL1-win)
GL1.hometeam <- 1:nrow(GL1)
GL1.STRdf<- GL1[,10]
GL1.HT <- GL1[,9]
GL1.win <- GL1[,2] == "H"
GL1.simple <- data.frame(id = GL1.hometeam, STRdf = GL1.STRdf,
                         HT = GL1.HT, win <- GL1.win)

# Create a data frame with test values covering the entire range
# of normalized petal lengths and normalized petal widths.
# This will allow a grid to be plotted of classified points 
# to show the decision boundary.
test <- expand.grid(STRdf=seq(min(GL1.simple[,2]), max(GL1.simple[,2]),
                           by=0.02),
                    HT=seq(min(GL1.simple[,3]), max(GL1.simple[,3]), 
                           by=0.02))
test2 <- data.frame(test)

# Loop to use kNN on the simplified GL1 data frame 
# with k from 1 to 15
for (k in 1:15) {
  # Predict the test points and calculate the probabilities
  GL1.pred.knn <- knn(GL1.simple[,2:3], test, GL1.simple[,4], k=k, prob=TRUE)

  # Obtain the probabilities from the prediction results
  prob <- attr(GL1.pred.knn , "prob")

  # Setup a data frame of class predictions (2-class)
  dataf <- bind_rows(
      mutate(test, prob=prob, cls="FALSE",
        prob_cls=ifelse(GL1.pred.knn==cls, 1, 0)),
      mutate(test, prob=prob, cls="TRUE",
        prob_cls=ifelse(GL1.pred.knn==cls, 1, 0)))

  # Plot the resulting test points, training points, and decision boundary
  thePlot <- ggplot(dataf) +
    geom_point(aes(x=STRdf, y=HT, col=win),
      data = mutate(test, win=GL1.pred.knn),
      size=1.2) + 
    geom_contour(aes(x=STRdf, y=HT, z=prob_cls, group=cls, color=cls),
      bins=2,
      data=dataf) +
    geom_point(aes(x=STRdf, y=HT, col=cls),
      size=3,
      data=data.frame(STRdf=GL1.simple[,2], HT=GL1.simple[,3], cls=GL1.win)) +
    ggtitle(paste("GL1 win kNN Model (k=", k, ") with Decision Boundary", sep="")) +
    theme(plot.title = element_text(face="bold", size=14, hjust=0.5)) +
    labs(x="Shoot Accuracy Diff (normalized)", y="Attendance(normalized)")

  # Show the plot
  print(thePlot)

  # Save the plot to a file
  filename <- paste("Lab9.knn", k, ".decisionboundary.png", sep="")
  png(filename, width=500, height=500)
  print(thePlot)
  dev.off()

  # Plot the resulting test points, training points, and decision boundary
  # Represent the point classification proabaility with point size
  thePlot <- ggplot(dataf) +
    geom_point(aes(x=STRdf, y=HT, col=win, size=prob),
      data = mutate(test, win=GL1.pred.knn)) + 
    scale_size(range=c(0.8, 2)) +
    geom_contour(aes(x=STRdf, y=HT, z=prob_cls, group=cls, color=cls),
      bins=2,
      data=dataf) +
    geom_point(aes(x=STRdf, y=HT, col=cls),
      size=3,
      data=data.frame(STRdf=GL1.simple[,2], HT=GL1.simple[,3], cls=GL1.win)) +
    geom_point(aes(x=STRdf, y=HT),
      size=3, shape=1,
      data=data.frame(STRdf=GL1.simple[,2], HT=GL1.simple[,3], cls=GL1.win)) +
    ggtitle(paste("GL1 win kNN Model (k=", k, ")\nwith Decision Boundary and Probability",
      sep="")) +
    theme(plot.title = element_text(face="bold", size=14, hjust=0.5)) +
    labs(x="Shoot Accuracy Diff (normalized)", y="Attendance (normalized)")

  # Show the plot
  print(thePlot)

  # Save the plot to a file
  filename <- paste("Lab9.knn", k, ".decisionboundaryprob.png", sep="")
  png(filename, width=500, height=500)
  print(thePlot)
  dev.off()
} 

#################################
## Classifying with Naive Bayes #
#################################
 
GL1.nb.model <- naiveBayes(FTR ~ .,data = GL1.train)
 
print(GL1.nb.model)
 
str(GL1.nb.model)
 
summary(GL1.nb.model)
 
GL1.nb.pred <- predict(GL1.nb.model, GL1.test)
 
eval_model(GL1.nb.pred, GL1.test)

################################################
## Decision Tree, Rule Set and Regression Tree #
################################################
 
# Build a decision tree for FTR using C4.5 (Weka's J48 implementation)
GL1.model.nom <- J48(FTR ~ ., data=GL1.train)

# View details of the constructed tree
summary(GL1.model.nom)

# Plot the decision tree
plot(GL1.model.nom)

# Create a regression tree for petal.length using rpart (CART implementation)
GL1.model.reg <- rpart(STRdf ~ ., data=GL1.train[,1:7])

# View details of the constructed tree
summary(GL1.model.reg)

# Plot the regression tree
plot(GL1.model.reg, uniform=TRUE,
     main="Regression Tree for GL1 Petal Length")
text(GL1.model.reg, use.n=TRUE, all=TRUE, cex=.8)

# Attempt post-pruning of the regression tree to see if a better
printcp(GL1.model.reg)

# Obtain one of the pruned trees using the cp value (round up)
GL1.model.reg.prune <- prune(GL1.model.reg, cp=0.010000)

# Verify it is the correct pruned tree (will be last listed)
printcp(GL1.model.reg.prune)

# Display the pruned tree (assuming the user chose to remove any nodes)
plot(GL1.model.reg.manualprune, uniform=TRUE,
     main="Regression Tree for Pruned GL1 Petal Length")
text(GL1.model.reg.manualprune, use.n=TRUE, all=TRUE, cex=.8)

#
# Build a Rule Set Using RIPPER
# Remember that RIPPER creates a default rule for the majority class
# and then creates rules to cover the other classes
#

# Build the rule set
GL1.model.rules <- JRip(FTR ~ ., data=GL1.train)

# Display the rule set
print(GL1.model.rules)

###############
## Evaluation #
###############

# Create predictions from the decision tree model using the test set
GL1.predict.nom <- predict(GL1.model.nom, GL1.test)

# Calculation of performance for nominal values uses a confusion matrix
# and related measures. 
GL1.eval.nom <- confusionMatrix(GL1.predict.nom, GL1.test$FTR)

# Display the evaluation results for the decision tree
print(GL1.eval.nom)

 
# Create predictions from the rule set using the test set
GL1.predict.rules <- predict(GL1.model.rules, GL1.test)

# Calculation of performance for nominal values uses a confusion matrix
# and related measures.
GL1.eval.rules <- confusionMatrix(GL1.predict.rules, GL1.test$FTR)

# Display the evaluation results for the rule set
print(GL1.eval.rules)

 