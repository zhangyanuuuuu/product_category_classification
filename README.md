# product_category_classification
product category classification from web images and text


## Usage

```bash
python classify.py
```

output is already generated as product_data_with_category.json
##

## Overall design
Three classifier, one based on image, two based on text. Then use majority vote for final result.

Image based: manually labelled first 30 images as training data, use VGG16 as feature extractor and cosine knn(1) for classification

Text based:
1. rule based: same manually labelled 30 images with 1,2,3-gram, if some gram only appears in 1 category, then assign that gram to the category. Then for new description, use majority vote or exclusive rule for prediction
2. word embedding based: use fast text embedding for the words, and find embedding similarity for category and words in description. Choose top 3 most similar words to represent each category and choose the most similar categoory.

## Q/A
- Why are you designing the solution in this way?

Information available are text and image. Hard part is no label, thus no training data for supervised learning. Fine tuning on NLP and CNN models still need data.

Word embedding approach is purely unsupervised, but since the training data is general, hard to have good result.

- What are the aspects that you considered when designing?

Purely unsupervised way likely cannot work well, since the task is to classify. So need some training data.

But I do not want to label too many data, since it should be an automatic method. Label 500 and predict the other 500 would be much accurate, but feel too onerous on human label.


- What are the cases your solution covers, how are they covered and why are they
important?

Categories with human labeled data are covered.


- What are the cases your solution does not cover and what are the ways you can
extend your current solution for them?

Some category does not exist in manually labeled data (first 30), so it is hard to predict them. More labelled data with fine tuning on NLP and CNN models will get better results.
I also considered using deep NLP models such as BERT, but likely cannot work well unless have more labeled data for fine tuning.


## License
[MIT](https://choosealicense.com/licenses/mit/)