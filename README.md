
# seq2seq Review Summary

Seq2Seq review summary with the [Amazon Reivews Dataset](http://times.cs.uiuc.edu/~wang296/Data/) ([direct download link](http://times.cs.uiuc.edu/~wang296/Data/LARA/Amazon/AmazonReviews.zip)).

## Setup

First, we need to put the data in the project root under data.  Symlinking is a great option:

```
# Symlinking option
$ ln -s ~/data/amazon-reviews-dataset data
$ ls data
cameras  laptops  mobilephone  tablets  TVs  video_surveillance
```

Data prep:

```
$ PYTHONPATH=$(pwd) python3 main.py prep
$ ls data/*.csv | xargs wc -l
    582046 data/x.csv
    582046 data/y_overall.csv
    582046 data/y_title.csv
   1746138 total
```
