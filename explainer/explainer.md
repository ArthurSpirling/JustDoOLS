# Why Simple Models Almost Always Work Best: Non-Technical Explainer

Our paper has some technical elements, though the underlying ideas are straightforward. Here we give a non-technical account of our claims and our evidence for those claims.  The intended audience is one of social scientists who are potentially unfamiliar with recent developments in machine learning---especially "deep learning". Below each section you will find a "too long, didn't read" summary. If you want something even shorter, our paper shows

1. that the intrinsic dimension of political *datasets* is low.  What we mean by this is that our data are fundamentally quite simple and have highly structured relationships between the variables.  This does not mean the "true" process by which data is produced in the real world is simple, but the datasets as they *arrive to us* are simple. As a result, complex machine learning models have nothing to "get hold of" and thus they don't do particularly well on our prediction problems. 
2. that this occurs due to something we call "data curation".  This is the process by which researchers gather limited numbers and types of variables in a way that have straightforward (if any) relationships with the outcome.
3. given 1 and 2, the way forward for machine learning in political science is more complex, raw and unstructured data like audio, video, text and images. 

## 1. Machine Learning and Prediction

Suppose you wanted to predict conflict between nations, or the winner of a general election, or the outcomes for children in the state social welfare system.  How would you do it?  First, let's be clear about what we mean by *predict*.  What we care about is good *forecasts*: you give me a bunch of conditions---values of variables (the *X*s)---and I tell you what will happen to the outcome (*Y*).  For instance, you tell me the state of the economy (*X*<sub>1</sub>), how popular the candidates are (*X*<sub>2</sub>), whether there are incumbents running (*X*<sub>3</sub>) and so on...and I tell you who will win (*Y*). Importantly, we are not trying to *explain* how candidates win votes with some theoretical model of the world which we then take to data.  We are just trying to do the best job we can---the most *accurate* job we can---of saying what will happen. So, for example, we don't think that one poll has a *causal* effect on another poll a month later.  But that earlier poll is probably *informative* about what will happen in that later poll and then in the election itself.

What methods would we use to *produce* the prediction?  The answer over the past 50 years or so has been *machine learning*.  Exactly what we mean by "machine learning" depends on the field and the problem, but the basic idea is to *automate* some of the model-building and model-fitting we would otherwise do by hand.  To be more specific, we are interested here in *supervised learning*. This is the situation where we ``train" a model based on pre-existing examples of a phenomenon---their inputs (the *X*s) and their outcomes (the *Y*s).  The "machine"---which we say more about below---will hopefully learn how the inputs are related to the outcomes.  The key, and the reason we use supervised learning, is because it can try all sorts of possible relationships between *X* and *Y* that we might never have thought of or imagined.  

That is, rather than us saying "try a model where the incumbent wins if the unemployment rate is low enough, and the GDP growth is high enough, and the US is not involved in any foreign wars", our supervised approach will try *that* model, and millions of other such models that all vary from each other.  That variation comes from different functions it is using to combine the variables, including multiplying them by each other, squaring them, cubing them and so on. And it will do all this *automatically* and in a way that optimizes the correctness of its predictions. You can get a sense of these combinations from the figure which shows a "neural network": the *X* variables enter on the left hand side and lots of different combinations are tried such that we can predict the target of our inquiry (the *Y*). 
<img align="right" src="DNN.png" width=280 title="Neural Net combines variables"> 
We can evaluate how well the machine has done by checking its accuracy on a set of data we kept back from the set of observations we let it use to learn the relationship between *X* and *Y*. For that ``held out" data, we give it the value of the inputs and ask for its best guess of the outcomes (*Y*)---which we know, but that do not show it.  If it does well enough on that *test set* of data, we can deploy it in the real world.  

So, for example, we might train an election predicting model on US presidential elections from the following post-war years: 1948, 1956, 1964, ..., 2012, 2020. That is, every second election.  The machine will consider all the variables we have at hand---the prevailing unemployment rate, the inflation rate of the election year, the Fed interest rate etc. It will use them to predict what happened in those elections---say, "did the Democrat win?".  Then we see how well that model performs on our *test* set: the *X*s and *Y*s for 1952, 1960, 1968, ..., 2008, 2016.  If we think it looks accurate enough, we plug in the *X*s for 2024, and see what it predicts. 
<img align="right" src="warming.jpeg" width=290 title="Predicting Warming"> 
Different researchers will use different machines that combine the data in different ways and thus they will have different predictions, but we can judge them by seeing which is closest to the truth---for example, one of the Global Warming Projections in the figure will be better than the others. 


**tl;dr: machine learning is about prediction. Machine learning models can automatically try all sorts of relationships between inputs and outcomes in an effort to forecast well---these include models we would never have thought of**

## 2. Prediction in Political Science

How is this working out?  Thinking broadly, machine learning has revolutionized science. We see it everywhere: from image analysis to geology to medicine to physics.  A special class of complicated models called (neural) nets has been particularly successful, and a related field of "[deep learning](https://www.nature.com/articles/nature14539)" has grown up around them. Many of the breakthroughs we see today in "Artificial Intelligence" and "Large Language Models" are based on those techniques.

What about in political science?  Here the news is less positive.  There have been many papers that use "machine learning" but it is typically in an *unsupervised* setting, like topic models.  There the researcher believes there is some hidden structure in the data---say, documents---and the machine tries to find that structure.  It is then the researcher's job to say what that structure is, exactly---what the clusters represent, or what the dimensions of debate are.  The goal is not *prediction* of anything.  

Where political and other social scientists *have* used machine learning for prediction, the results have been mixed. This is perhaps best exemplified in a 2020 paper by [Salganik et al](https://www.pnas.org/doi/10.1073/pnas.1915006117).  For that paper, many different teams tried to predict life outcomes for children in a large data set.  Two results stood out: 

1. none of the predictions were very good. That is, none of the teams were able to predict what *actually* happened to the children very well. 
2. no one could beat simple models. That is, despite lots of fancy machine learning models being tried, simple approaches like linear regression or logistic regression did best.

This is surprising: we would imagine that the process by which children move through their lives and have various good or bad outcomes is very complicated.  So we would expect models that can cope with potentially very complicated processes would do well---or at least better than simple models. But this general pattern---best predictions aren't very accurate, and simple models do just as well---turns up in lots of political science papers.  The puzzle is *why*.  

Notice that nothing here relies on the argument that complex models are hard to *interpret* relative to e.g. a linear regression. This is certainly true, and can be a [real concern](https://www.nature.com/articles/s42256-019-0048-x).  But our work proceeds from the more fundamental observation that complex models just don't do very well in the first place. 

**tl:dr: machine learning has been applied to complex political and social processes, but it has struggled: models are often not very accurate, and very simple models often do as well as very complicated ones.**

## 3. Intrinsic Dimension of Political Science Data is Low

Our argument is that political science data has low **intrinsic dimension**.  Basically this means that political science datasets do not have the kind of subtle variation and non-linearities that machine learning models can take advantage of over and above simple models.  This does **not** mean that politics is a simple process: rather, the data *as it arrives to us as researchers* is simple in structure.  

### Reducible Problems 
To see the problem, let's return to  predicting an election.  This might be a complicated problem because the way a particular type of person votes may depend on their demographic characteristics, plus their economic situation, plus whether they were in a good mood on election day due to the weather etc.  A machine learning model could in principle try many many combinations of these variables, including interactions ("this 45 year old Princeton professor is more likely to vote for the incumbent if the sun is shining, but this 44 year old NYU one is not").  You would need a lot of *parameters* (like the <em>&beta;</em>s in a regression)---one for Princeton, one for professor, one for good weather, one for Princeton-professor, one for  Princeton-professor-weather etc etc.  But that's ok, because we're using machine learning. But our point is that a lot of political science data does **not** have this sort of high quality, fine grained data to which to fit these many parameters.  So it doesn't make much difference whether you use a complex model or not.  Another way to express the problem is that political science modeling problems are highly **reducible**.  This means that they may initially look very complicated, but in practice you don't need much information to do a pretty good job of predicting what you are trying to predict. 


An example of this phenomenon would be something like predicting the outcome of a UK House of Commons vote in the 1950s. 
<img align="right" src="commons55.jpg" width=240 title="House of Commons: Govt v Oppn"> 
There were around 650 members of parliament, and they were almost all either in the Labour party or the Conservative party (see the picture where each dot is a seat after the 1955 election). One of those parties (always) formed the Government, and one of them  (always) formed the Opposition.  Whichever way the government voted, the opposition voted against. Suppose you were trying to predict whether a given bill would pass the House.  What would you need to know?  We might initially think we need the "ideal points" of all the 650 MPs, and where they lie relative to what the bill proposes.  But you don't: you only need to know how the leader of the Government is voting, and perhaps the leader of the Opposition.  Once you know that, you know how everyone in the governing party, and the chamber, will vote.  So we *reduced* a problem of needing 650 parameters down to one of needing only 2.  And we were able to do this because there was much more strucure in the data than it initially appeared. 

### Finding the Intrinsic Dimension

How can we know exactly what the intrinsic dimension of a dataset is?  The basic idea is to fit a very complicated model with lots of parameters (looking for lots of 'effects'), and then try a model with fewer parameters (looking for fewer 'effects'), and keep reducing the number.  When you get to a low enough parameter number that the model starts fiting poorly---meaning it doesn't predict very well relative to the model with many many parameters, you have found the intrinsic dimension.  Unsurprisingly, the details are more technical than this, and involve something called **random subspace training** invented by [Li et al (2018)](https://arxiv.org/abs/1804.08838). And in practice, one fits an arbitrarily complex model called a [deep neural network](https://news.mit.edu/2017/explained-neural-networks-deep-learning-0414) to predict the outcome.  As we mentioned above, these models combine the variables available to them in complex ways that make for accurate predictions in many fields.  Importantly, we can compare across models with different numbers of parameters easily: for instance, a model with 10000 parameters is "twice" as complex as one with 5000, which is "twice" as complex as one with 2500 and so on. 
<img align="right" src="cifar_results.jpg" width=120 title="Intrinsic Dimension of CIFAR">
### CIFAR as a Baseline


So what do we see? Well first, we need a baseline to compare results to.  We use the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset for that.  This consists of 60000 images in 10 classes.  Those classes are things like airplanes, cars, cats, dogs, horses and so on.  


The idea is to train machine learning models on 50000 of the images (their pixels), and try to correctly predict the airplanes, cars, cats, dogs etc in the test set images.  This is a famous and classic problem, and we know from other work that the returns to "big" models---i.e. models with lots of parameters---tend to be excellent.   Reassuringly, that is what we found too.  We started with a deep neural net that had millions of parameters.  It scores very highly on the "usual" performance metrics like [accuracy](https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers#Single_metrics), [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall). Then we slowly "back off" the model, reducing the number of parameters available to it to fit the data.  

And the model responds as we would expect---reduced complexity reduces how well it fits, from a very high baseline down to quite poor performance (e.g. accuracy below 10%, when we make only 2500 parameters---exp(8)---or so available). The column figure to the right makes this point: from top to bottom we have accuracy, precision, recall, area under the curve.  The x-axis is the intrinsic dimension (log scale).  

### Political Science Data
What about for political science data?  Does show a similar return to complex models?  In a word: **no**.  

We looked at a set of datasets which have appeared in our top journals in the context of machine learning models.  They were: 
-  **American Politics**: the original data is from [Blackwell (2013)](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-5907.2012.00626.x), though our source was [Montgomery & Olivella (2018)](https://onlinelibrary.wiley.com/doi/10.1111/ajps.12361). The problem is to predict "whether a Democratic candidate has "gone negative" in a given week" of a campaign, for both incumbents and non-incumbents in US elections.
-  **Political Instability**: the original data is from [Goldstone et al (2010)](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1540-5907.2009.00426.x). The problem is to forecast political instability, which can assume many forms including "the onsets of both violent civil wars and nonviolent democratic reversals".
-  **International Conflict**: the original data is from [Beck et al (2000)](https://www.jstor.org/stable/pdf/2586378.pdf).  The problem is to predict a binary variable coded as 1 if a state is "engaged in an international
conflict, 0 if it is at peace." 
- **Civil War**: the original data is from [Muchlinski et al](https://www.jstor.org/stable/24573207), though our source was [Colaresi & Mahmood (2017)](https://journals.sagepub.com/doi/full/10.1177/0022343316682065).  The problem is to "forecast the binary observation of the presence or absence of a civil war onset". 
- **Fragile Families** from Salganik et al (2020), as described above.  There are three (binary) targets to predict: (household) Eviction, (primary caregiver) job training, (primary caregiver) layoff.
 
Starting on the left with the CIFAR data, here are our results: 

<p>
<img align="center" src="main_results.jpg" width=800 title="Political Science Intrinsic Dimension"> 
</p>

What we see is that **(a) complex machine learning models---i.e. those with lots of parameters---often cannot do very well at all**. For instance, the rightmost three columns from Salganik et al suggest that even very complex neural nets struggle to do better than 80% accuracy on certain problems.  The second thing we see is that **(b) to the extent complex models *can* do well, so can simple ones.**  For example, for the American politics, instability and IR conflict data, topline accuracy is acheived with quite basic models---with say exp(4)=55 or exp(6)=400 numbers of parameters.

**tl:dr: political science datasets have low intrinsic dimension; they look fundamentally different to datasets where complex models are able to extract big performance improvements**

## 4. Why is Intrinsic Dimension so low?  Data Curation.

In the second half of the paper we argue that political science datasets are inhospitable for complex machine learning because of the way they are gathered.  In particular, we think that researchers have cognitive and financial limits when they put together *X* variables to predict some target *Y*.  That is, it is simply costly---in money terms or in terms of thinking about all the various possible causes and correlates---to get the sort of data for which machine learning "works" so well.  

One specific point we make here is that, in contrast to something like the image prediction problem above, political scientists typically have **tabular data**.  That is, data where the rows are observations, and the columns are various (often highly coarsened, censored, truncated) covariates recorded for those observations.  For example, a dataset of 50 states might have some rows and some columns that look like this:

| State  | turnout04 | unemployment | percent foreign born |
| ------------- | ------------- | ------------- | ------------- | 
| Nebraska | 58.02  | 3.8  | 7.4|
| New Hampshire  | 71.6  | 3.7   | 6.4|
| New Jersey  | 59.92 | 4.8   | 23.4|

But the types of data where machine learning does well are *non* tabular cases where information is stored in a more basic "raw" format, not in columns and rows.  For example, audio or video data has this form. 

In our paper, we give some technical results where we think about a case where *Y* (the outcome) is a product of some variables **X** and **Z**.  In particular, the "true" situation is 


<p align="center">
Y =  f(X) + g(Z)+ error
</p>

but the researcher does not know the functions *f* or *g*.  We go through various scenarios, but the simplest one is where **X** and **Z** are uncorrelated.  In this case, the researcher can find themselves in a situation where they gather data about **X**, fit a potentially simple model to it (the *f* part) and never realize they are missing **Z**.  It doesn't matter how much they increase the complexity of *f* (i.e. how deep the neural net they try), they can never do better than a simple model even though the *true* data generating process that gave them the values of *Y* they saw was very complex.

<img align="right" src="xy3d.jpg" width=200 title="uncurated Data">
A toy example may drive this logic home.  Suppose that the "true" data generating process is this (complicated) function: 


$$
Y = 1 + 0.2x_1 + 
\frac{1}{ 1+ \exp( -6( \frac{x_1\times x_2}{\sqrt{2}} )  - \frac{1}{3}) } + \epsilon
$$


If we plot *Y* and *X*<sub>1</sub> and  *X*<sub>2</sub> we get something like the colorful plot to the right.  But the problem is that the researcher is unaware of  *X*<sub>2</sub> or doesn't care about it, or can't afford to collect it.  In that case, they end up modeling *Y* as a function of *X*<sub>1</sub> *only*.  Now their plot looks like the 2D figure.  They may suspect something non-linear is going on, but actually a linear regression works "fine" insofar as it gives rise to a statistically significant predictor, and the model fit (the *R*<sup>2</sup>) is quite high by social science prediction standards (25%).
<img align="right" src="xy_linear.jpg" width=250 title="Curated Data">

### A Typology of Problems

In the paper we provide a typology of types of machine learning---that is, data---problems that one can encounter in practice.  The two axes of interest are, first, how "inherently" or "truly" difficult the modeling processes *really* is.  Here we mean, essentially, how convoluted and complex the function that takes us from inputs to outcomes is (the Data Generating Process, or DGP). The second axis is how much noise we can expect when dealing with this type of problem.  By "noise" we mean anything in the data that makes it hard to uncover the true relationship between *X* and *Y* (whether it is truly simple or complex); this could include the usual random variability, but we also mean it in a broader sense to include e.g. variables we would like to have access to but do not, perhaps because they are expensive to obtain.


|               |   simple DGP            | complex DGP  | 
| ------------- | ------------- | ------------- | 
| **Low Noise** | (a) Fundamental Relations   | (b) Learnable Problems |
| **High Noise**  |  (d) Curated Data |   (c) Unlearnable Problems |

There are four broad scenarios here: 
- (a) Fundamental Relations: these are scenarios where simple models perform well, and there is no "extra" return to complex ones. Basic physics or engineering systems have this form.  For example, predicting the boiling point of water based on altitude.  This relationship is well known, easy to accurately predict and is linear (the boiling point falls by a constant amount with every 1000 feet up we go).
- (b) Learnable Problems: these are scenarios where simple models do poorly, but complex ones can make progress.  Problems that have been solved using complex models on high dimensional "raw" data have this form.  For example, using neural nets to learn whether a photo is of a cat or a dog.
- (c) Unlearnable Problems: these are scenarios where the  DGP is very complicated, but noise is very high: for example, predicting someone's exact happiness or dating success on a given day.  This is something that involves very hard to measure quantities, but is also just inherently hard to model---not least because there are strategic/interactive effects of relationships with other agents (observations). Complex models do slightly better than simple ones, but neither does especially well.
- (d) **Curated Data**: this is the scenario we think is most common in political science.  Researchers "start out" theoretically with a problem that is truly in box (c): predicting war or unemployment or sports results or something else very difficult.  But they gather limited, highly coarsened data that pushes them over to box (d), but keeps the noise approximately the same.  So no models do well, and complex models make no improvements over simple ones (there is nothing for the complex models to do).

  



**tl;dr: political science datasets are highly curated.  Thus complex machine learning models cannot improve over simpler options**

## 5. So What?  Advice to Practitioners

Supposing we are correct---that data is often highly curated, and this means minimal payoff from machine learning in political science---so what?  What follows from this in terms of *positive* advice?  We have four suggestions:
1. **Know your problem** We think that, too often, political scientists don't know much about the data in front of them---its origins, how and when it was constructed and so on.  But for the reasons we explained above, this structure and curation affects how likely complex models are to be useful.  
2. **Start simple** The first move, for most political science data, should *not* be some state of the art, highly complex model.  For one thing, these may not be very interpretable, but more fundamentally they probably won't work.  And by "won't work" we mean "do anything more impressive than a (generalized) linear model".  Starting with something simple gives a proper baseline for your problem, and may allow you to get more of a sense of how to ramp up the complexity. 
3. **Do Diagnostics** Here we mean, essentially, using techniques like the ones we have explored above on your own data.  In a relatively short period of time you can estimate the intrinsic dimension and compare it to that of our focus datasets above---including the image data, where complex models are helpful.  This may save some effort in modeling choice terms. 
4. **Get More and Better Data** Very generally, if the idea is to get accurate predictions and you do not think the system is a simple one (and we do not think social systems are), you need much more data.  This is partly about having more observations, but it's also about have more covariates and "raw" data to predict from.  Getting media like text, audio, video may help---and they do need more complex models to be included. 
