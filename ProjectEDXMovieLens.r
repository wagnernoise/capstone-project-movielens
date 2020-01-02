################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
     semi_join(edx, by = "movieId") %>%
     semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#EDA
##Dimensions and properties
glimpse(edx)
glimpse(validation)


##Number of different movies in the dataset
qplot(edx$rating, 
      geom = "histogram", 
      xlab = "Rating", 
      ylab = "Frequency",
      fill = I("blue"))


edx %>% group_by(rating) %>%
  summarise(number = n()) %>%
  arrange(desc(number)) %>%
  top_n(5)


##Movie with greatest number of ratings

edx %>% group_by(title) %>%
  select(rating) %>%
  summarise(number = n()) %>%
  arrange(desc(number))



#Genres by rating
movies_by_rating <- edx %>% separate_rows(genres, sep = "\\|") %>% 
  group_by(genres) %>%
  summarise(number = n()) %>%
  arrange(desc(number))
movies_by_rating


#Bar plotting the results 

movies_by_rating %>% 
  mutate(genres = reorder(genres, number)) %>%
  ggplot(aes(genres, number, fill = genres)) +
  geom_bar(width = 0.5, stat = "identity") +
  coord_flip() +
  theme(legend.key.height = unit(0.8,"line"))


#Word cloud of the results
library(wordcloud)
movies_by_rating_2 <- edx %>% 
  separate_rows(genres, sep = "\\|") %>% 
  group_by(genres) %>% 
  summarize(Ratings_perGenre_Sum = n(), 
            Ratings_perGenre_Mean = mean(rating), 
            Movies_perGenre_Sum = n_distinct(movieId), 
            Users_perGenre_Sum = n_distinct(userId))

wordcloud(words = movies_by_rating_2$genres, freq = movies_by_rating_2$Ratings_perGenre_Sum, 
          min.freq = 10, max.words = 10, random.order = FALSE, random.color = FALSE, 
          rot.per = 0.35, scale = c(5, 0.2), font = 4, colors = brewer.pal(8, "Dark2"), 
          main = "Most rated genres")


percentage.mistake.genres <- (8188/sum(movies_by_rating$number))*100
cat("Percentage of contributions that are misclassified:", percentage.mistake.genres, "%")

##Movies per year
#library(lubridate)
movies_year <- edx %>%
  extract(title, c("title", "year"), regex = "^(.*) \\(([0-9 \\-]*)\\)$", remove = F)

movies_per_year <- movies_year %>%
  select(movieId, year) %>% # select columns we need
  group_by(year) %>% # group by year
  summarise(count = n())  %>% # count movies per year
  arrange(year)

head(movies_per_year,10)

movies_per_year_df <- as.data.frame(movies_per_year)
  
movies_per_year_df %>%
  ggplot(aes(x = as.numeric(year), y = count)) +
  geom_line(color="blue") +
  xlab("Year") +
  ylab("Movies per year")


movies_per_year$year[which.max(movies_per_year$count)]

#Classification models: 
## Model 1: Average rating

#Splitting the edx dataset into train and test
split_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
edx_train <- edx[-split_index,]
edx_test <- edx[split_index,]

# Root mean squared error
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


### Building the Recommendation System

mu_hat <- mean(edx$rating)
mu_hat

#Prediction using meand rating value
naive_rmse <- RMSE(validation$rating, mu_hat)
naive_rmse

#Prediction using only values of 2.5 as rating
predictions <- rep(2.5, nrow(validation))
less_than_mean <- RMSE(validation$rating, predictions)

#Table of RMSE results
#All results will be shown at the end
rmse_results <- tibble(method = c("Just the average", "Lower than average (2.5)"), RMSE = c(naive_rmse, less_than_mean))


## Model 2: Item-based and user-based collabirative systems

#Getting the mean ratings value
mu <- mean(edx$rating) 

#Item-based model  
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

model_1_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie Effect Model",
                                 RMSE = model_1_rmse ))

#Item and user-based model
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_2_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))


## Model 3: Regularization 
#Item-based regularization
mu <- mean(edx$rating)
lambdas <- seq(0, 10, 0.25)

just_the_sum <- edx %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

rmses_movies <- sapply(lambdas, function(l){
  predicted_ratings_reg1 <- validation %>%
    left_join(just_the_sum, by='movieId') %>%
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(validation$rating, predicted_ratings_reg1))
})

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie Effect Model",  
                                 RMSE = min(rmses_movies)))

#Item and user-based regularization
rmses_movies_user <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings_reg2 <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(validation$rating, predicted_ratings_reg2))
})

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie + User Effect Model",  
                                 RMSE = min(rmses_movies_user)))

## Model 4: Matrix factorization
library(recosystem)
train_data <- data_memory(user_index = edx_train$userId, 
                          item_index = edx_train$movieId, 
                          rating = edx_train$rating, 
                          index1 = T)

test_data <- data_memory(user_index = edx_test$userId, 
                         item_index = edx_test$movieId, 
                         rating = edx_test$rating, 
                         index1 = T)
rec <- Reco()
rec$train(train_data, 
          opts = c(dim = 30, 
                   costp_l2 = 0.1, 
                   costq_l2 = 0.1, 
                   lrate = 0.1, 
                   niter = 100, 
                   nthread = 4, 
                   verbose = F)
) 

predictions_mf <- rec$predict(test_data, out_memory())

model_mf_rmse <- RMSE(edx_test$rating, predictions_mf)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Matrix factorizartion Model",  
                                     RMSE = model_mf_rmse
                          )
)
# Arraging in descending order
rmse_results %>% arrange(desc(RMSE))


