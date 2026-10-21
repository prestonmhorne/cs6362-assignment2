OpenReview
.net

Login
back arrowGo to ICLR 2017 homepage
Understanding deep learning requires rethinking generalization 
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals
Published: 06 Feb 2017, Last Modified: 12 Oct 2025ICLR 2017 OralReaders:  EveryoneShow BibtexShow Revisions
TL;DR: Through extensive systematic experiments, we show how the traditional approaches fail to explain why large neural networks generalize well in practice, and why understanding deep learning requires rethinking generalization.
Abstract: Despite their massive size, successful deep artificial neural networks can
exhibit a remarkably small difference between training and test performance.
Conventional wisdom attributes small generalization error either to properties
of the model family, or to the regularization techniques used during training.

Through extensive systematic experiments, we show how these traditional
approaches fail to explain why large neural networks generalize well in
practice. Specifically, our experiments establish that state-of-the-art
convolutional networks for image classification trained with stochastic
gradient methods easily fit a random labeling of the training data. This
phenomenon is qualitatively unaffected by explicit regularization, and occurs
even if we replace the true images by completely unstructured random noise. We
corroborate these experimental findings with a theoretical construction
showing that simple depth two neural networks already have perfect finite
sample expressivity as soon as the number of parameters exceeds the
number of data points as it usually does in practice.

We interpret our experimental findings by comparison with traditional models.
Keywords: Deep learning
Conflicts: mit.edu, zju.edu.cn, google.com, berkeley.edu
Community Implementations: CatalyzeX 12 code implementations
Reply Type:all
Author:everybody
Visible To:all readers
Hidden From:nobody
53 Replies
[–]
Deep models fit and generalise significantly less confident when more random labels exist 
Xinshao Wang
03 Jul 2022 at 17:16 (modified: 04 Jul 2022 at 05:29)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: We present a new insightful finding to complement this one “**deep neural networks easily fit random labels** (Understanding deep learning requires rethinking, Zhang et al., ICLR 2017)”: **Deep models fit and generalise significantly less confident when more random labels exist**.

Therefore, in summary, deep machine learning is quite consistent with human learning:
(1) When a trainee is given noisy (e.g., wrong or biased) supervision, it will fit noise (e.g., error or bias).
(2) When the supervision and guidance contain more noise, the trainee will learn less confidently.

Please read more details: 
arXiv: https://arxiv.org/abs/2207.00118
biorxiv: https://www.biorxiv.org/content/10.1101/2022.07.01.498447v1.full.pdf
[–]
public review by 
[Deleted]
ICLR 2017 conference public reviewReaders:  EveryoneShow Revisions
[–]
Review after five years: excellent paper inspires many years' DL study. 
(anonymous)
21 May 2022 at 00:25ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: After five years, when we look at this paper we still find a lot of interesting observations.
This paper's contributions can be concluded:
1. This paper presents an important problem that has not been resolved even after five years. 
Previous generalization literature cannot faithfully explain the behavior of deep neural networks. 
What should be the core of generalization theory of deep learning?
2. This paper summarizes interesting experimental findings of deep neural networks.
Deep neural networks may fit the training labels easily, while suffering from a huge generalization gap.
However, overfitting issues are not dominant in these days [1]. How to better understand overfitting?
3. This paper illustrates the diffilculty of building a general theory of deep learning.
The generalization bound can become trivial under random cases, and the algorithmic illustrations are data-independent.
I think the deep learning is too broad a method that we cannot say many general things without knowing the specific tasks.

[1] 2022-ICLR-how does vit work
Commented by: 电光幻影炼金术《从审稿到中稿》栏目
[–]
public comment by Oriol Vinyals 
Oriol Vinyals
26 May 2022 at 04:36ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Thank you, this is very kind. And agreed! If you are interested, we have a "retrospective" on this paper published at ACM:

"Understanding deep learning (still) requires rethinking generalization"

https://cacm.acm.org/magazines/2021/3/250713-understanding-deep-learning-still-requires-rethinking-generalization/fulltext
[–]
The resistance is expected 
Srichakradhar Reddy Nagireddy
12 Jun 2020 at 22:33ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: The authors have decided to challenge the hard fixed notion in the deep learning community about the generalizability of a network, which is the most basic component of a deep neural network. The results they have produced have shaken my pre-existing beliefs and perceptions on the neural networks.
It would have been better if they had pointed to a specific example of where the model could classify with random label. I would like to research more about the implications of random label classification and memorization in models.
Even though a huge model may be able to memorize the data / features, I feel it would not make sense to build a model with wasted capacity to learn random labels.
The discovery that the dropout generalization also doesn't help much is what grabbed my attention.
Overall, it was a great paper challenging a lot of existing beliefs and notions taken for granted in the deep learning realm. I strongly hope it  encourages / challenges all researchers to come up with a better researach to defend the neural architectures.
[–]
About defending the neural networks, how to leverage self knowledge, and how to learn confidently with self correction. 
Xinshao Wang
08 Jul 2022 at 14:02ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Our recent work could be interesting to you, about defending the neural networks, how to leverage self knowledge, and how to learn confidently with self correction. 
/forum?id=Sy8gdB9xx&noteId=fiLGxY5EssW
[–]
A very disappointing paper 
Thomas G Dietterich
29 Jan 2018 at 23:59ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: I expected to like this paper, because I respect the authors, and many people have said good things about it. In addition, I enthusiastically support experimental work to improve our understanding of how deep nets work. I'm sorry to say I was very disappointed.

I agree with Pirmin Lemberger that the results in this paper are completely unsurprising. I'm surprised that the authors were surprised. I'm shocked that at least one reviewer thought this was ground breaking.

Suppose you replace every occurrence of "neural net" in this paper with "decision tree". Would you still be surprised? Of course trees would be able to memorize the data. Of course the trees would need to be deeper for random labels than for random pixels (because random pixels provide more information for memorizing the data; shuffling the original pixels is just another way of generating random pixels). Of course regularizing the depth of the trees would not make them generalize well. The only difference is that trees would perform very poorly on the original data too, because they lack the right representations to generalize well.

Regularization only prevents overfitting. It does not ensure that there are good hypotheses in the hypothesis space. Your experiments with randomly parameterized convolutional layers show that they DO help ensure that there are good hypotheses in H. 

I note in Figure 1 that it takes 5000 steps to fit the original data, and more than 13000 steps to fit random labels. This directly contradicts your statement that training did not slow down substantially. Training time nearly tripled.

It has long been known that the most effective way to control hypothesis complexity in neural networks is early stopping. Back in the 90s, the standard procedure was to use the biggest hidden layer you could afford, and then do early stopping. (IIRC this appears in the "Tricks of the trade" book https://books.google.com/books?isbn=3540494308). 

In your experiment, if we apply early stopping at 5000 steps, we will get a non-trivial Radamacher complexity for this data set. On random labels, the accuracy will be near zero. Did you really not notice this?

I think it is interesting that you did not need to change your hyperparameters to get the network to fit random labels. That suggests that the hyperparameters might depend more on the architecture and not so much on the data. Have people seen this in other cases?
[–]
An introduction to generalization and regularization in DL for non experts 
Pirmin Lemberger
02 Apr 2017 at 03:25 (modified: 06 Apr 2017 at 02:05)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: The paper looks very interesting and thought provoking to me. However, having an education as a theoretical physics I found it hard at times to follow through all the arguments in the present forum. To clarify my own thoughts about this fascinating topic and to share them with data scientists I work with I wrote a short pedagogical paper arxiv.org/abs/1704.01312 , trying to connect everyday practice of data scientist practice with some basic theoretical knowledge in statistical learning, especially Rademacher complexity. Perhaps this could be helpful to other people without advanced theoretical knowledge in statistical learning. The focus is on explicit mathematical definitions and on a discussion of relevant concepts.
[–]
pending question 
(anonymous)
27 Mar 2017 at 16:45ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Could the authors report on ImageNet the training accuracy using both AlexNet and Inception with random labels and data augmentation ? Does that work? and do hyper-parameters need to change?

This has been requested several times in the discussion, but it has never been addressed (only on CIFAR10). 
Only this experiment may fully substantiate the general claims made by the authors.

Thank you.
[–]
Thought-provoking, but claims of significance not sufficiently backed up 
Matthias Minderer
16 Mar 2017 at 13:53 (modified: 16 Mar 2017 at 13:54)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: The authors claim to have shown that "the traditional approaches fail to explain why large neural networks generalize well in practice". 

While the paper was generally thought-provoking to read and discuss, the paper did not make it clear to me what these "traditional approaches" are and why it is surprising that they fail to explain the performance of neural networks.

The authors express their surprise at various abilities of neural networks, but consistently fail to provide references or arguments beyond "conventional wisdom" that explain why I should be surprised. A few examples:

"To our surprise, several properties of the training process for multiple standard achitectures is largely unaffected by this transformation of the labels. This poses a conceptual challenge. Whatever justification we had for expecting a small generalization error to begin with must no longer apply to the case of random labels."

What is the conceptual challenge? It is not obvious to me that it is impossible to reconcile small generalization error with with the ability to fit noise. A reference or reasoning for being surprised would help.

"Surprisingly, stochastic gradient descent with unchanged hyperparameter settings can optimize the weights to fit to random labels perfectly, even though the random labels completely destroy the relationship between images and labels."

Why is this surprising? We know that the network has more than enough parameters to memorize the random labels, so it is not obvious that this is surprising.

"As shown in the last three rows of Table 2 in the appendix, although it does not reach the perfect 100% top-1 accuracy, 95.20% accuracy is still very surprising for a million random labels from 1000 categories."
"Quite surprisingly, fitting the training labels exactly yields excellent performance for convex models."
"Surprisingly, adding regularization does not improve either model’s performance!"

Again, why are these findings surprising? Is there any quantitative reasoning behind any of them?

Perhaps I lack the background in statistical learning theory that may be necessary develop the intuitions that lead the authors to find their results surprising. Consequently, I think the paper would benefit from more rigorous arguments for why the findings are surprising.
[–]
Not useful 
Xiang Zhang
22 Feb 2017 at 13:23 (modified: 22 Feb 2017 at 17:21)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: (I read this paper after knowing that it got accepted to ICLR 2017)

I am quite happy that a paper that actually talks about model complexity can be accepted to ICLR 2017. But I want to express my personal opinion (unrelated to my lab, my advisor etc) that this paper is a bit over-rated. Calling this paper "groundbreaking" is a bit ridiculous in my honest opinion, and it is unfair to the theory community. I hold no bad ideas to the authors or the chairs, but only to the technical details in both experiments and theory of the paper.

TL;DR: The paper shows that generalization is bad for a problem with random labels. That is of course true, but uninteresting.

In section 2.2 2nd paragraph, the claim that the empirical Radamacher complexity is 1 is a bit hand-waving. One detail that is worth noting (but this does not mean the paper is wrong) is that the conclusion assumes that all functions in the hypothesis set H is bounded by 1 in absolute values. This is only true if H is defined as the set of error functions, or if H is the set hypothesis functions but only for a binary classification problem. The experiments with random labels seem to be conducted with multi-class classification. It would be non-trivial to conclude that the complexity is 1 in this case when H is defined as the set of error functions for multi-class classification problem, albeit it seems the result would be true to me (sorry for my hand-waving here).

However, Rademacher complexity can also be used to bound the loss (or energy in the energy-based models framework [1]), which is the objective that our algorithm truly optimizes. There are papers that use Rademacher complexity this way. Optimizing a multi-class classification problem will decrease the difference between the upper- and lower-bounds of the loss, therefore decreasing the loss Rademacher complexity, but in very distinct manners between correct label case and random label case. Therefore, the experiments on random labels may not describe anything about loss Rademacher complexity in correct labels.

Note that the previous paragraph linked Rademacher complexity to the dynamics of optimization, not just the hypothesis set. This is because at any stage of optimization (such as at a step of SGD), the set of hypotheses that the algorithm can explore afterwards is limited relative the current hypothesis that the algorithm possesses. There is an evolving upper-bound that the algorithm can never jump out of, because it needs to "minimize". This evolving hypothesis set is called "sublevel hypothesis set" in [2], which is the reason why the actual Rademacher complexity is evolving with the optimization process, rather than staying static as assumed by this paper or a majority of theoretical papers from the community. Note that all this applies to either correct labels or random labels.

Thus far, I think practitioners in deep learning can safely ignore the possible implication from the article that your problem is not bounded in generalization -- that is not yet proven to be true, because the paper misses the piece of ingredient that can combine the dynamics of optimization with complexity measurement, and another ingredient that characterizes the difference between the random-label problem and the correct-label problem to make the result mean anything. These ingradients are discussed in previous research, therefore I think the title about "rethinking generalization" is a bit unfair to other people who have contributed to the theory literature.

Below is a bit "hand-waving" on the distinction between random-label problem and correct-label problem. There is one more theorem that I want to mention to motivate more discussions on the difference of optimization between correct labels and random labels. It is the Talagrand's contraction lemma ([3], also a perhaps easier-to-understand version in [2]). It can be used to bound Rademacher complexity R(F(G)) with L * R(G), in which F(G) are the set of composite functions h(g) with h in H and g in G, and L is the maximum possible Liptchitz constant of functions in H.

Note that "deep" learning models are essentially function compositions. Talagrand's lemma therefore means that the quality of your deep learning model's generalization bound depends on whether the function compositions have a Liptchitz constant smaller than 1 (that is, "contracting"). The conjecture I hope somebody can take a further look at is that, with correct labels, at the final optimization stage you can easily get contracting function compositions, but not with random labels, because the former assumes certain smoothness in some semantic space where classification makes sense.

[1] LeCun, Yann, et al. A tutorial on energy-based learning. Predicting structured data 1 (2006): 0.
[2] Zhang X. PAC-learning for energy-based models (Master dissertation, Courant Institute of Mathematical Sciences, New York, 2013).
[3] M. Ledoux and M. Talagrand. Probability in Banach Spaces: Isoperimetry and Processes. Springer, 1991.
[–]
A possible explanation for generalization not fully considered in the paper? 
(anonymous)
25 Feb 2017 at 20:31 (modified: 25 Feb 2017 at 20:47)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: I think the main message of the paper is not to show "that generalization is bad for a problem with random labels", but rather it shows training accuracy can be as good for a problem with randomized labels. While it is interesting and thought-provoking, I also think it is not super surprising, or groundbreaking. I could be wrong, but it seems the authors did not fully debunk a explaining theory (If they do, please definitely comment and let me know!) that: implicit regularization by SGD plays important role in selection of generalizable models over their counterparts. Namely, with the same cost, models with smaller norm are easy to "reach" under SGD training, and that's why current SGD based training would produce generalizable models when they can also memorize.
[–]
ICLR committee final decision 
ICLR 2017 pcs
06 Feb 2017 at 09:54ICLR 2017 conference acceptanceReaders:  EveryoneShow Revisions
Comment: The authors report the experimental findings of a fascinating inquiry on the ability of the deep neural networks to fit randomly labelled data. The investigation is sound, enlightening, and inspiring. The authors propose both a) a theoretical example showing that a simple shallow network with a number of parameters large enough wrt sample size yields perfect finite-sample expressivity; b) a systematic extensive experimental evaluation to support the findings and claims. The experimental evaluation is a model of thoroughness. 
 
 This is definitely groundbreaking work, which will inspire many works in the coming years.
Decision: Accept (Oral)
[–]
A few questions regarding section 5 
(anonymous)
26 Jan 2017 at 23:17ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Very inspiring paper. But I got a few questions regarding section 5 and hope that you can help clarify

(1) You showed that the solutions to the linear model solved by SGD lie in the span of the data points, given that the initial point is zero. But this is true not only for SGD, but also for many other first order methods, right? All the analyses related to SGD in this section would hold for many other first order methods as well. So this set of experiments can hardly support your point that SGD acts as implicit regularizer.

(2) You claimed that SGD would often converge to the solution with minimum norm, but I think this happened here mostly because the initial point was set to zero. Any local minimization method would converge to a local minimum that is close to zero(minimum norm solution) if it starts from zero. I do agree SGD may implicitly regularize a model, but I reckon the effect of implicit regularization shown in this set of experiments was probably/mostly because of choosing zero as initial points.


[–]
Can it be well learned with large norm initialization? 
(anonymous)
25 Feb 2017 at 20:45ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: If the nets cannot be well trained using initialization with norm similarly to ones under good convergence, then the argument (that SGD implicitly regularizes l2 norm) holds, as the norm increases during the training. So the real question is: can it be well learned with large norm initialization? It seems rarely the case in practice?
[–]
SGD: implicit regularization via flat-minima 
David Krueger
30 Jan 2017 at 23:52ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: From what I've heard, SGD implicitly regularizes by finding flatter minima, see, e.g.:
http://cims.nyu.edu/~achoroma/NonFlash/Papers/Entropy-SGD.pdf
https://papers.nips.cc/paper/899-simplifying-neural-nets-by-discovering-flat-minima.pdf 

[–]
What was observed in the experiments in Section 5 was probably not due to 'SGD implictly regularizes'.
(anonymous)
05 Feb 2017 at 22:20 (modified: 05 Feb 2017 at 22:22)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Thank you for the references. David. I wasn't questioning 'SGD implicitly regularizes'. My point was, what was observed in the experiments in Section 5 was not due to 'SGD implictly regularizes', but simply because the initial point was set to zero which led to a kind of 'least norm solution/explicit regularization'. 
[–]
What is the point of section 5? 
(anonymous)
25 Jan 2017 at 11:29ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: I cannot see how section 5 relates to the rest of the paper.

You say: "it is useful to appeal to the simple case of linear models to see if there are parallel insights that can help
us better understand neural networks." but I do not see any relevant insights in this section.

The claim seems to be that SGD does some implicit regularization.
It has been demonstrated previously that SGD implicitly regularizes by finding "flatter" minima.

You seem to suggest that SGD might also regularize implicitly by finding a minimum l2-norm solution, but then your results suggest that this is not the case, because:
1. l2-regularization only gives a small improvement
2. wavelet pre-processing increases performance but increases l2-norm

[–]
Universality Theorem 
(anonymous)
23 Jan 2017 at 01:13 (modified: 23 Jan 2017 at 01:13)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: According to the universality theorem (UT), (shallow) neural networks can approximate any function. In paper, you showed neural networks can fit a randomly labeled dataset. Don't we expect this phenomenon based on UT? 
[–]
I'd say 'no' for two reasons 
Moritz Hardt
24 Jan 2017 at 17:50ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: 1. Universality theorems don't say anything about optimization. For example, our paper gives a very simple finite sample universality theorem showing that for any labeling of the function there *exists* a shallow neural net that can fit that labeling. This theoretical result doesn't mean that we can efficiently find such a labeling using stochastic gradient descent. This is however what the experiments show.

2. The fact that there exist neural networks that can fit any labeling of the data doesn't by itself imply that natural architectures used in practice can do so. But this is again something our experiments imply.
[–]
Kernel Solution? 
(anonymous)
22 Jan 2017 at 04:24 (modified: 22 Jan 2017 at 04:38)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Others have pointed out this in comments. But, I'm confused about the kernel solution.

The authors mentioned that using the kernel method, we can fit any labeling for MNIST. For actual labeling, they got 1.2% test error, and stated that "this kernel solution has an appealing interpretation in terms of implicit regularization. Simple algebra reveals that it is equivalent to the minimum L2-norm solution of Xw = y."

Does it mean that the minimum L2-norm solution for MNIST yields 1.2% test error? As I understand, Xw = y is not to be solved in the kernel feature space.
[–]
public comment by 
[Deleted]
ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
[–]
Linear Models -- what is going on? 
(anonymous)
20 Jan 2017 at 14:51ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: I am totally confused by the section on linear models.

For MNIST, d << n, so it seems you *cannot* fit any labelling.
But as I understand, you claim to do so, and to achieve 1.2% test error with the result.

Can you please clarify this?
[–]
Kernels is the answer : ) 
Oriol Vinyals
20 Jan 2017 at 17:31ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Sure: the model is linear in the "lifted" feature space f(x) where f is a basis of the RKHS defined by the RBF kernel. This is called "kernelization" -- there are some good books out there explaining these methods in detail. 
[–]
Please clarify this section 
(anonymous)
25 Jan 2017 at 11:19 (modified: 25 Jan 2017 at 11:21)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: You should mention this in the text.  You do not say that you use a kernel, or that it is the RBF kernel.

Without specifying otherwise, X should mean the data in the pixel-space.  This is also confusing because when you use the RBF kernel, d = infinity.

Also, when you say: "the l2-norm of the minimum norm solution with no preprocessing is approximately 220" this is in the RBF-kernel feature space, correct?  How does one compute the (approximate) norm in this space?
[–]
Implicit Regularization 
Behnam Neyshabur
20 Jan 2017 at 10:24 (modified: 20 Jan 2017 at 10:26)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: This is an important discussion and very significant issue.  It's worth noting our ICLR 2015 paper that already raised this issue and argued, based on different experiments, that generalization is coming from the optimization dynamics rather then a capacity bound on the model class, including discussion of explicit and implicit regularization and an analogy to matrix factorization:

https://arxiv.org/pdf/1412.6614v4.pdf

But even though the issue was raised and discussed before, the experiments in the submission are indeed an excellent illustration of how high the capacity of neural networks are, and drives home the point that generalization is coming from a completely different place.
[–]
References 
(anonymous)
20 Jan 2017 at 21:43ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Thanks for pointing this out. We have updated this missing reference in the paper.
[–]
Response to comments and reviews 
(anonymous)
10 Jan 2017 at 20:30ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: We thank the reviewers for their careful reading of our paper and many helpful questions and suggestions.

With regards to Reviewer 1’s first question, we believe that this is indeed the most important open question in machine learning.  We intend on pursuing exactly this question in future work.  

Both Reviewer 2 and a public commenter asked for some clarification for what we mean by random labels.  Specifically, we take the same set of classes as in the original problem.  For each example, we assign to it a random class label.  For example, for CIFAR 10, there are ten classes.  For each example, we choose a random integer between 1 and 10, and assign that example to be in the class indexed by our random integer.  This assignment remains consistent across epochs throughout training.

Reviewer 3 requested some clarification of our experiments with kernels.  Indeed, we used a one-versus all classification scheme.  For each class a target vector was constructed with a 1 if the example was in the class and a -1 if the example was not in the class.  Hence, there was a different weight vector for each class.

One public comment asked about our MNIST experiments.  We clarify that the minimum l2 norm w which satisfies Xw=y does indeed achieve a test error of 1.2%.

We would also like to respond an additional insightful public comment about data augmentation.   Understanding the benefits of data augmentation is important, but it is not cut and dry as to how to count the number of examples after augmentation.  Since each augmented example is highly correlated  with an existing training example, it is not clear how to measure the size of the augmented data set.  However, we have been able to get zero training error on augmented versions of CIFAR-10 (including crops, rotations, and flips) using the Inception architecture.   Regardless, we note that the conceptual problems raised by our experiments are not resolved whether or not data augmentation is taken into account.

Finally, another excellent public comment raised the issue that non-parametric classifiers can memorize labels.  We fully agree.  That popular neural net architectures behave like non-parametric classifiers is what we are trying to argue in the paper and is somewhat surprising.  Note that it is easy to construct a neural net that cannot memorize---the simplest example being linear regression with a rank-deficient design matrix.  Moreover, the comment suggests that capacity of nonparametric classifiers can be controlled by regularization.  We show in our experiments that the amount of regularization used in practice is not sufficient to control the capacity of neural net models.  
[–]
loose ends 
(anonymous)
12 Jan 2017 at 08:04 (modified: 12 Jan 2017 at 16:43)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Please allow me to insist : you didn't answer my concerns on the missing experiments on ImageNet with random labels.


- You said : (comment on Dec 16th 2016)
"For fitting random labels, we only studied random cropping augmentation on cifar10 as reported in the paper. We think investigating other jitter augmentation is an interesting question that merits future study."

- You said : (parent comment)
"However, we have been able to get zero training error on augmented versions of CIFAR-10 (including crops, rotations, and flips) using the Inception architecture.".

- The numbers for cifar10 are nowhere to be found in the paper, yet there is : (abstract of the paper)
"convolutional networks [...] easily fit a random labeling of the training data. This phenomenon is qualitatively unaffected by explicit regularization"

- Data augmentation is included in "explicit regularization" as the paper says : (Sec 1.1, paragraph "The role of explicit regularization.")
"We show that explicit forms of regularization, such as weight decay, dropout, and data augmentation, [...]."


**** Question 1) Regarding the experiments that you did on augmented versions of CIFAR-10, can you please give the details on the corresponding setup for reproducing your findings ?

I would like to check, in particular, how the number of parameters compares to the total number of augmented samples (upper bounding the size of the augmented dataset, which you say should be not clear how to measure).

You said you obtained zero training error in that case.
Can you please update Table 1 with the missing numbers ?


**** Question 2) You say : (Section 2.1)
"We also tested random labels on the ImageNet dataset [...]
The network also manages to reach 90% top-1 accuracy even with explicit regularizers turned on."

I would like your confirmation that I understood section 2.1 correctly: did you manage to get a similar result on ImageNet (low training error with data augmentation and random labels) ?
How much did you get ?

Can you please update Table 2 with the missing numbers ?


**** Question 3) You say : (Section 1.1)
"Our central finding can be summarized as: Deep neural networks easily fit random labels."
This paper studies a form of overfitting. Standard practice (since Krizhevsky 2012) consists in using data augmentation against overfitting. However I see that you chose to exclude data augmentation from some experiments.

Or did you decide that data augmentation was irrelevant for some experiments ? If yes then why ? Can you discuss this in the paper clearly ?

Can you please point out which of your results are still true when this explicit regularizer is activated in your setups ?


[–]
Response 
(anonymous)
20 Jan 2017 at 20:14ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: We updated the PDF and added an appendix E to include the results on fitting random labels with explicit regularizers (including data augmentation). 



[–]
Minimum L2 norm solution Xw=y 
(anonymous)
11 Jan 2017 at 21:51 (modified: 11 Jan 2017 at 22:09)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: I have two questions about the statement: "the minimum L2 norm w which satisfies Xw=y does indeed achieve a test error of 1.2%."

1- Do you imply that the training data of MNIST are linearly separable? i.e., there exists a "w", for which Xw=y, where X is training data and y the corresponding labels.
2- If we can get test error of 1.2% by minimum L2 norm solution, why do we bother using more complex classifiers, such as neural networks?
[–]
Response 
(anonymous)
20 Jan 2017 at 18:48ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Yes, they are linearly separable in the feature space associated with the kernel. MNIST is an easy dataset and commonly used as demonstration or sanity check in the deep learning community.
[–]
What is the kernel? 
(anonymous)
02 May 2017 at 10:07ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: All questions are related to Section 5.

1.
Regarding your comment "linearly separable in the feature space associated with the kernel":

Please forgive if this should be obvious from the paper but what is the kernel associated with the experiment that gives 1.2% error rate on the test set?  The paper specifically mentions "on MNIST with no pre-processing".  To me this sounds like you are using the raw pixels as features and that the training set is "linearly separable" without qualifying with "in the feature space associated with the kernel."  What do I misunderstand here?

2.
In section 5. you derive mathematically that for the separable case the linear model obtained with SGD optimization provides the solution with the minimum L2 norm of the weight vector.  Is this the right interpretation of the result?  If yes then why did you not do this experiment using SGD, which would easily fit into the memory of a much lower-spec computer?

3.
From another reply, you are training 10 separate linear models, one for each class.  Are you assigning the class labels during testing as:  class=argmax_i (model_i)?

[–]
Question 
Greg Yang
10 Jan 2017 at 18:40ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Congratulations on an excellent paper!

I have a question that I hope one of the authors can humor me. I'm not sure I understand the claim made in the implications section. For example, the following appears in the paragraph *Rademacher complexity and VC-dimension*.

> Deep neural networks with more than n parameters have VC-dimension greater than n, which leads to trivial uniform convergence bounds.

If we replace "deep neural networks" with, say, "linear thresholds", the premise of this statement still holds, but the clause following "which" is false, as standard VC arguments show that linear thresholds are PAC learnable. More precisely, the class of linear threshold functions on R^d has d+1 parameters and VC dimension d+1, and by (essentially) the Sauer-Shelah Lemma, the number of samples required to learn linear thresholds to within error E and with (1 - D) confidence is O(E^-1 log D^-1 + d E^-1 log E^-1), which is a nontrivial "uniform convergence bound." (Kearns and Vazirani 1994)

Could the authors clarify what is meant in the above statement and perhaps present their arguments more explicitly for the reader?

[1] Kearns, Michael, and Umesh Vazirani. 1994. An Introduction to Computational Learning Theory.
[–]
Response 
(anonymous)
20 Jan 2017 at 18:34ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: The bound mentioned here is n = O(d/E). If one solves for E = O(d/n), the conclusion is then clear.
[–]
Still confused 
Greg Yang
20 Jan 2017 at 23:32ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Correct me if I'm wrong. It seems to me that you are saying the following:
"Because our empirical experiments indicate that NNs with more than n parameters can learn any function on a fixed sample of size n (i.e. shatters the sample), the bound given by VC theory is uninformative because it predicts O(d/E) samples, where E < 1/n, whereas we used only n << d/E samples to learn the target function on the size n sample."

If this is indeed your argument, then it mistakes sample error for global error, which is what VC theory and PAC learning is concerned with. The bound given in my previous post is for learning a function making at most error E under arbitrary *global distribution*, not just on the sampled points. Your experiments show that NNs with more than n parameters indeed have VC dimension greater than n, but in order to learn the correct function with small error globally, you need many more samples, the number of which is given in my last post.

If this is not your argument, I apologize, and please elaborate what you mean.
[–]
response 
(anonymous)
21 Jan 2017 at 00:10 (modified: 21 Jan 2017 at 00:11)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Yes, absolutely. If you *fix the hypothesis space* and *let n goes to infinity*, then learnability is guaranteed provided finite VC-dimension. 
[–]
comment 
Greg Yang
21 Jan 2017 at 10:11ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: We agree on that. So I'm assuming you disagree with my interpretation of your comment? Could you formulate your argument in the paper precisely? (For example, state clearly in your claim whether you are fixing the hypothesis space, whether n goes to infinity, what do you fix as a constant, etc)
View 4 More Replies →
[–]
Is test accuracy of the Kernel method really ~99%? 
(anonymous)
09 Jan 2017 at 17:44ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Hi,

In paper, you derived a kernel solution XX^T a = y, and then mentioned that "On MNIST with no preprocessing, we are able to achieve a test error of 1.2% by simply solving (3)." Is it a typo that you achieved test error of ~1%? Isn't it the training error? 

You then mentioned that "Note that this kernel solution has an appealing interpretation in terms of implicit regularization. Simple algebra reveals that it is equivalent to the minimum L2-norm solution of Xw = y". So, one may wonder how "minimum L2-norm solution" yields ~99% accuracy on test data?!

Thanks.
[–]
Yes. 
Moritz Hardt
24 Jan 2017 at 17:58ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: We mean *test* error, not training error. Even Yann LeCun's page on MNIST results shows that kernel methods achieve test error roughly in this regime. So, this shouldn't strike you as too surprising. These days the entire kernel matrix fits in main memory on a large machine thus making optimization much easier than back when people reported results for MNIST using kernel methods.
[–]
A classic result of non-parametric methods? 
(anonymous)
09 Jan 2017 at 08:12ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Dear authors,

Any non-parametric classifier (with number of parameters = O(number of data)) can memorize random labels. For instance, one can reproduce the result in this submission with 1-nearest neighbour and random forest classifiers in less than 10 lines of Python. Do 1-nearest neighbour and random forest classifiers "require rethinking generalization"? It is unsurprising that the Rademacher complexity of non-parametric classifiers is constant, since their number of parameters grows with data, and their capacity is controlled via regularization.

```
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.neighbors import KNeighborsClassifier as kn
import numpy as np

np.random.seed(0)

n = 1000
x = np.random.randn(n,224*224)  # random ImageNET-sized images
y = np.random.randint(2,size=n) # random labels

print(rf(n_estimators=n).fit(x,y).score(x,y)) # training accuracy = 1.0
print(kn(n_neighbors=1).fit(x,y).score(x,y))    # training accuracy = 1.0
```
[–]
Some questions about random labels 
(anonymous)
22 Dec 2016 at 17:12ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: -Could you please clarify how did you do the "random labels" modification to the input labels? are "all" labels from a given class mapped to the "same" random label? (more specifically, two examples have the same label before modification, will they have same random label after the modification) or a label is replaced with a random one regardless of its original label (i.e. two examples have same labels before modifications, will likely get two different random labels)? 
- Let's take cifar10 as an example. The original data set has 10 classes, after the "random label" modification how many classes are there? 10? >10? <10? 

[–]
official review by AnonReviewer3 
ICLR 2017 conference AnonReviewer3
20 Dec 2016 at 21:45ICLR 2017 conference official reviewReaders:  EveryoneShow Revisions
Review: This paper offers a very interesting empirical observation regarding the memorization capacity of current large deep convolutional networks. It shows they are able to perfectly memorize full training-set input-to-label mapping, even with random labels (i.e. when label has been rendered independent of input), using the same architecture and hyper-parameters as used for training with correct labels, except for a longer time to convergence. 
Extensive experiments support the main argument of the paper. 

Reflexions and observations about finite-sample expressivity and implicit regularization with linear models fit logically within the main theme and are equally thought-provoking.

While this work doesn’t propose much explanations for the good generalization abilities of what it clearly established as overparameterized models,
it does compel the reader to think about the generalization problem from a different angle than how it is traditionally understood.

In my view, raising good questions and pointing to apparent paradox is the initial spark that can lead to fundamental progress in understanding. So even without providing any clear answers, I think this work is a very valuable contribution to research in the field.

Detailed question: in your solving of Eq. 3 for MNIST and CIFAR10, did you use integer y class targets, or a binary one-versus all approach yielding 10 discriminant functions (hence a different alpha vector for each class)?
Rating: 10: Top 5% of accepted papers, seminal paper
Confidence: 4: The reviewer is confident but not absolutely certain that the evaluation is correct
[–]
official review by AnonReviewer2 
ICLR 2017 conference AnonReviewer2
18 Dec 2016 at 14:29ICLR 2017 conference official reviewReaders:  EveryoneShow Revisions
Review: This paper presents a set of experiments where via clever use of randomization and noise addition authors demonstrate the enormous modeling (memorization) power of the deep neural networks with large enough capacity.  Yet these same models have very good generalization behavior even when all obvious explicit or implicit regularizers are removed.  These observations are used to argue that classical theory (VC dimension, Rademacher complexity, uniform stability) is not able to explain the generalization behavior of deep neural networks, necessitating novel theory.

This is a very interesting and thought provoking body of work and I am in complete accord with the observations and conclusions of the paper.  The classical generalization theory is indeed often at a loss with complex enough model families.  As the authors point out, once model families reach a point where they have capacity to memorize train sets, the classical theory does not yield useful results that could give insight into generalization behavior of these models, leaving one to empirical studies and observations.

A minor clarification comment: On page 2, “ … true labels were replaced by random labels.”  Please state that random labels came from the same set as the true labels, to clarify the experiment.
Rating: 9: Top 15% of accepted papers, strong accept
Confidence: 3: The reviewer is fairly confident that the evaluation is correct
[–]
Memorization, overfitting, generalization 
ICLR 2017 conference AnonReviewer1
13 Dec 2016 at 14:49 (modified: 13 Dec 2016 at 14:49)ICLR 2017 conference official reviewReaders:  EveryoneShow Revisions
Review: the authors of this work shed light on the generalization properties of deep neural networks. Specifically, the consider various regularization methods (data augmentation, weight decay, and dropout). They also show that quality of the labels, namely label noise also significantly affects the generalization ability of the network.

There are a number of experimental results, most of which are intuitive. Here are some specific questions that were not addressed in the paper:
1. Given two different DNN architectures with the same number of parameters, why do certain architectures generalize better than others? In other words, is it enough to consider only the size (# of parameters) of the network and the size of the input (number of samples and their dimensionality), to be able to reason about the generalization properties of a given network?

2. Does it make sense to study the stability of predictions given added dropout during inference?

Finally, provided a number of experiments and results, the authors do not draw a conclusion or offer a strong insight into what is going on with generalization in DNNs or how to proceed forward. 
Rating: 10: Top 5% of accepted papers, seminal paper
Confidence: 4: The reviewer is confident but not absolutely certain that the evaluation is correct
[–]
Data augmentation + random labels 
(anonymous)
09 Dec 2016 at 13:03 (modified: 09 Dec 2016 at 13:03)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Can you still fit random labels on the training set when using jittering data augmentation ?
[–]
response 
(anonymous)
16 Dec 2016 at 07:26ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: For fitting random labels, we only studied random cropping augmentation on cifar10 as reported in the paper. We think investigating other jitter augmentation is an interesting question that merits future study.
[–]
where ? 
(anonymous)
10 Jan 2017 at 05:40 (modified: 10 Jan 2017 at 05:41)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Hi,

I can't find the section where you studied random cropping augmentation on cifar10 for fitting random labels.
I don't see the numbers in Table 1. Can you add them ?
[–]
follow-up 
(anonymous)
16 Dec 2016 at 15:01ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: I think there is something deeper here and a result of major importance is missing, that could make the paper significantly stronger.

---

I am glad that you ran the ImageNet experiment with AlexNet because it raises the following issue :
- I believe [Krizhevsky et al. 2012] sets the standard for "practice" in training neural networks and I hope we all agree on this. This paper uses data augmentation as part of the training scheme, which means that the number of data points evaluated by the neural network is 2048 * 1.2 million = over 2 billion.
=> AlexNet has 60 million parameters. 
=> Data augmentation is used to exactly combat overfitting, which in some sense is what this paper seems to analyze.
=> If my understanding is correct, in practice the number of data points far exceeds the number of parameters.
=> This fact seems to contradict what is in the abstract : "as soon as the number of parameters exceeds the number of data points *as it usually does in practice*" (emphasis added) so I am afraid it might be a little misleading for readers, and I hope the authors can either point where I am wrong and/or clarify this issue in the paper.

---

The abstract says that "Specifically, our experiments establish that state-of-the-art convolutional networks for image classification trained with stochastic gradient methods easily fit a random labeling of the training data. This phenomenon is qualitatively unaffected by explicit regularization [...]"

One of the most spectacular results in this paper is that AlexNet can learn a random labeling of ImageNet. (This result was notably shown in a slide of a presentation from one of the authors of the paper, which made a bit of a splash.)

However, I noticed in the appendix that in the Table 2, the experiment "ImageNet 1000 classes with random labels" was never run with data augmentation, which is the most popular explicit regularization scheme. This table looks incomplete to me.

I am very eager to see the result of this experiment because I believe it would make the paper much stronger and substantiate the claims much better.

---

It is my opinion that either the claims should be weakened or this result needs to be included. My hope is that the setup built by the authors should be ready to run this experiment and the paper can be amended accordingly by the acceptance deadline.
[–]
What about the optimizaiton objective? And finite-limited-precision? 
ICLR 2017 conference AnonReviewer3
03 Dec 2016 at 23:15 (modified: 03 Dec 2016 at 23:15)ICLR 2017 conference questionReaders:  EveryoneShow Revisions
Question: 1) Your experiments consider classification error, but the loss being optimized is always a different surrogate loss (e.g. conditional log likelihood). Did you look at the effect on the actual training objective being optimized? Can you comment on that?

2) Regarding the result on effective finite sample expressivity. This results relies on the ability to uniquely identify each of your n d-dimensional training data points, after projection, as a single unique real number. Isn’t this feasible primarily because they are « true » real values. Would it not break at some point with finite-limited-precision values?
[–]
Response 
(anonymous)
13 Dec 2016 at 02:17 (modified: 13 Dec 2016 at 02:17)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: 1) Yes, we monitored both the objective function (cross entropy) and the accuracy. The training objective in all cases converges to a near-zero quantity.  The cross entropy on the test set tracks the training error and then curves upward.  However, this increase in the test cross entropy is because the model becomes more confident about labels it has labeled incorrectly.  

2) For simplicity, we assumed infinite precision in our construction.  In finite precision, one could repeat the construction O(log(n)) times to avoid collisions.
About OpenReview
Contact
FAQ
Hosting a Venue
Sponsors
Terms of Use / Privacy Policy
All Venues
Donate
News
OpenReview is a long-term project to advance science through improved peer review with legal nonprofit status. We gratefully acknowledge the support of the OpenReview Sponsors. © 2026 OpenReview


OpenReview
.net

Login
back arrowGo to ICLR 2017 homepage
On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima 
Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, Ping Tak Peter Tang
Published: 06 Feb 2017, Last Modified: 12 Oct 2025ICLR 2017 OralReaders:  EveryoneShow BibtexShow Revisions
TL;DR: We present numerical evidence for the argument that if deep networks are trained using large (mini-)batches, they converge to sharp minimizers, and these minimizers have poor generalization properties. 
Abstract: The stochastic gradient descent (SGD) method and its variants are algorithms of choice for many Deep Learning tasks. These methods operate in a small-batch regime wherein a fraction of the training data,  say -- data points, is sampled to compute an approximation to the gradient. It has been observed in practice that when using a larger batch there is a  degradation in the quality of the model, as measured by its ability to generalize. We investigate the cause for this generalization drop in the large-batch regime and present numerical evidence that supports the view that large-batch methods tend to converge to sharp minimizers of the training and testing functions---and as is well known, sharp minima lead to poorer generalization. In contrast, small-batch methods consistently converge to flat minimizers, and our experiments support a commonly held view that this is due to the inherent noise in the gradient estimation. We  discuss several  strategies to attempt to help large-batch methods eliminate this generalization gap.
Keywords: Deep learning, Optimization
Conflicts: northwestern.edu, intel.com, ibm.com
Community Implementations: CatalyzeX 7 code implementations
Reply Type:all
Author:everybody
Visible To:all readers
Hidden From:nobody
15 Replies
[–]
Typo in Figure 5 
Amir H Abdi
08 Oct 2019 at 05:05ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: The values of epsilon in Figure 5 are set as: 1e-3 and 5e-3. 
Based on the previous experiments, and the fact that sharpness of 5e-3 values are less than sharpness of 1e-3 values, it is my understanding that 5e-3 should be changed to 5e-4. 
[–]
Potential mistake in Appendix C 
Amir H Abdi
08 Oct 2019 at 05:03ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: In the "PERFORMANCE MODEL" appendix, f_l(p) is assumed to be 1.0; however, P (number of processors) is assumed to be smaller than B_l. Therefore, the following statement does not match the provided intuition: "the parallel efficiency of the LB method, is 1.0. In other words, we assume that the LB method is perfectly scalable due to use of a large batch size"
[–]
ICLR committee final decision 
ICLR 2017 pcs
06 Feb 2017 at 09:52ICLR 2017 conference acceptanceReaders:  EveryoneShow Revisions
Comment: All reviews (including the public one) were extremely positive, and this sheds light on a universal engineering issue that arises in fitting non-convex models. I think the community will benefit a lot from the insights here.
Decision: Accept (Oral)
[–]
Analysis of large batch training 
ICLR 2017 conference AnonReviewer1
20 Dec 2016 at 10:30ICLR 2017 conference official reviewReaders:  EveryoneShow Revisions
Review: Interesting paper, definitely provides value to the community by discussing why large batch gradient descent does not work too well
Rating: 8: Top 50% of accepted papers, clear accept
Confidence: 3: The reviewer is fairly confident that the evaluation is correct
[–]
Thank you for your review 
Nitish Shirish Keskar
27 Dec 2016 at 13:54ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: We thank you for your review and your kind assessment.
[–]
Good paper 
Alex Lamb
17 Dec 2016 at 12:17ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: I think that this is a great empirical exploration of long-held folk wisdom in the Deep Learning community - that using larger minibatches makes generalization error worse.  

The paper does a good of explaining why phenomenon occurs, by analyzing the "sharpness" of the loss function for large-batch and small-batch trained models.  

Some other connections that I think would be interesting to explore: 
  -If you use very small minibatches, does generalization get even better (perhaps at the expense of very slow training).  
  -Can other forms of noise injection compensate for the use of a larger minibatch?  For example, if I inject increasing amounts of noise into the gradients or the parameters with larger batches, does this close the gap with small-batch training?  
  -The motivation for using smaller minibatches here seems closely related to the motivation for adversarial examples (ensuring that loss is relatively flat in a large region around data points).  
[–]
Clarifications 
Nitish Shirish Keskar
27 Dec 2016 at 13:56ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Thank you Alex for your interesting questions. 
From our preliminary experiments, it seems that there is not significant benefit from reducing batch-sizes to a very small value. For instance, in C1 and C2 networks, a batch-size of 8 or 16 led to (statistically) similar values of testing accuracy as a batch size of 256.

LB methods may very well benefit from noise injection, but we have not figured out how to do so successfully. We experimented with additive random Gaussian noise (both in gradients and in iterates), noisy labels and noisy input-data. However, despite significant tuning of the hyperparameters of the random noise, we did not observe any consistent improvements in testing error. 

Also, we agree that the motivation for using smaller minibatches is related to the motivation for adversarial examples. 
[–]
Little novelty but valuable empirical evidence 
ICLR 2017 conference AnonReviewer2
16 Dec 2016 at 17:57ICLR 2017 conference official reviewReaders:  EveryoneShow Revisions
Review: The paper is an empirical study to justify that: 1. SGD with smaller batch sizes converges to flatter minima, 2. flatter minima have better generalization ability. 

Pros and Cons:
Although there is little novelty in the paper, I think the work is of great value in shedding light into some interesting questions around generalization of deep networks. 

Significance:
I think such results may have impact on both theory and practice, respectively by suggesting what assumptions are legitimate for real scenarios for building new theories, or be used heuristically to develop new algorithms with generalization by smart manipulation of mini-batch sizes.

Comments:
Earlier I had some concern about the correctness of a claim made by the authors, which is resolved now. They had claimed their proposed sharpness criterion is scale invariance. They took care of it by removing this claim in the revised version.



Rating: 10: Top 5% of accepted papers, seminal paper
Confidence: 3: The reviewer is fairly confident that the evaluation is correct
[–]
Thank you 
Nitish Shirish Keskar
27 Dec 2016 at 13:55ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Thanks for your review and your suggestions on how our work could be used in further investigations.
[–]
Good paper 
ICLR 2017 conference AnonReviewer3
16 Dec 2016 at 17:26ICLR 2017 conference official reviewReaders:  EveryoneShow Revisions
Review: I think that the paper is quite interesting and useful. 
It might benefit from additional investigations, e.g., by adding some rescaled Gaussian noise to gradients during the LB regime one can get advantages of the SB regime.
Rating: 6: Marginally above acceptance threshold
Confidence: 4: The reviewer is confident but not absolutely certain that the evaluation is correct
[–]
Clarification regarding noise 
Nitish Shirish Keskar
27 Dec 2016 at 14:03 (modified: 27 Dec 2016 at 14:03)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Thank you for your review. We experimented with additive random Gaussian noise (both in gradients and in iterates), noisy labels and noisy input-data. However, despite significant tuning of the hyperparameters of the random noise, we did not observe any consistent improvements in testing error.  Overall, our feeling is that this needs deeper investigation and that LB methods may need to be modified in a more fundamental way to achieve good generalization.
[–]
local minima 
ICLR 2017 conference AnonReviewer3
02 Dec 2016 at 18:15 (modified: 02 Dec 2016 at 18:15)ICLR 2017 conference questionReaders:  EveryoneShow Revisions
Question: true "local minima" or just "local minima" of one-dimensional slices of a unimodal (within the search boundary) nonconvex problem?
[–]
Second-Order Optimality 
Nitish Shirish Keskar
07 Dec 2016 at 13:21 (modified: 07 Dec 2016 at 13:21)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Agreed, we cannot guarantee that the solutions obtained by the SB and LB methods are indeed local minima of the problem (as we did not attempt to verify second-order optimality). To be cautious, we will use the terms “LB solution” and “SB solution”, as appropriate, in our updated manuscript.
[–]
Contant "1" in equation (3)? 
ICLR 2017 conference AnonReviewer2
02 Dec 2016 at 16:18 (modified: 02 Dec 2016 at 16:18)ICLR 2017 conference questionReaders:  EveryoneShow Revisions
Question: If I understand correctly, the paragraphs above Eq (3) claim the sharpness criterion is scale invariant. Does that happen when the denominator in Eq (3) is "1+f(x)". f(x) could have any scale/unit but "1" is a unitless constant. How the proposed criterion is scale invariant with "1+f(x)"? Specifically, if I replace the objective function f(x) with a scaled version c*f(x), does your criterion produces the same value?
[–]
On The Sharpness Metric 
Nitish Shirish Keskar
07 Dec 2016 at 11:42 (modified: 07 Dec 2016 at 11:42)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: You are right, this metric is not scale invariant, and we will remove that statement from the paper in our next update. The inclusion of the “1+” in the denominator is of course to guard against the case when f(x) is zero, and as such it is useful for our purposes. Nevertheless, in response to your comment, we experiment with the denominator changed to f(x0) - f(x), where x0 is the initial point, and observed similar order-of-magnitude results, indicating that our numbers reported in the paper are not misleading.
About OpenReview
Contact
FAQ
Hosting a Venue
Sponsors
Terms of Use / Privacy Policy
All Venues
Donate
News
OpenReview is a long-term project to advance science through improved peer review with legal nonprofit status. We gratefully acknowledge the support of the OpenReview Sponsors. © 2026 OpenReview


OpenReview
.net

Login
back arrowGo to ICLR 2017 homepage
beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework 
Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, Alexander Lerchner
Published: 06 Feb 2017, Last Modified: 05 May 2023ICLR 2017 PosterReaders:  EveryoneShow BibtexShow Revisions
TL;DR: We introduce beta-VAE, a new state-of-the-art framework for automated discovery of interpretable factorised latent representations from raw image data in a completely unsupervised manner.
Abstract: Learning an interpretable factorised representation of the independent data generative factors of the world without supervision is an important precursor for the development of artificial intelligence that is able to learn and reason in the same way that humans do. We introduce beta-VAE, a new state-of-the-art framework for automated discovery of interpretable factorised latent representations from raw image data in a completely unsupervised manner. Our approach is a modification of the variational autoencoder (VAE) framework. We introduce an adjustable hyperparameter beta that balances latent channel capacity and independence constraints with reconstruction accuracy. We demonstrate that beta-VAE with appropriately tuned beta > 1 qualitatively outperforms VAE (beta = 1), as well as state of the art unsupervised (InfoGAN) and semi-supervised (DC-IGN) approaches to disentangled factor learning on a variety of datasets (celebA, faces and chairs). Furthermore, we devise a protocol to quantitatively compare the degree of disentanglement learnt by different models, and show that our approach also significantly outperforms all baselines quantitatively. Unlike InfoGAN, beta-VAE is stable to train, makes few assumptions about the data and relies on tuning a single hyperparameter, which can be directly optimised through a hyper parameter search using weakly labelled data or through heuristic visual inspection for purely unsupervised data.
Conflicts: google.com
Reply Type:all
Author:everybody
Visible To:all readers
Hidden From:nobody
14 Replies
[–]
ICLR committee final decision 
ICLR 2017 pcs
06 Feb 2017 at 09:54ICLR 2017 conference acceptanceReaders:  EveryoneShow Revisions
Comment: This paper proposes a modification of the variational ELBO in encourage 'disentangled' representations, and proposes a measure of disentanglement. The main idea and is presented clearly enough and explored through experiments. This whole area still seems a little bit conceptually confused, but by proposing concrete metrics and methods, this paper makes several original contributions.
Decision: Accept (Poster)
[–]
Asking very important questions of non-linear PCA (learning latent manifolds of datasets), often way beyond even human capabilities. Offering a toy solution, partly superseded by existing approaches. Non-scalable to real-life, non-toy data-sets.
Galin Georgiev
06 Jan 2017 at 20:41 (modified: 06 Jan 2017 at 20:56)ICLR 2017 conference public reviewReaders:  EveryoneShow Revisions
Review: The paper is attacking a classical and very hard problem: non-linear PCA i.e. learning the principal components a.k.a independent factors, a.k.a. orthogonal geodesics in the highly non-linear latent manifold of a given data-set. Moreover, the paper is hoping for these factors to be humanly-interpretable. The problem is so hard that is likely unsolvable in unsupervised fashion for real-life datasets. After all, even we humans are able to zero-in the the type of, say, legs of chairs (Fig 3 in the paper), only after we have identified the object as a chair and have rotated and translated it. The importance of this problem is hard to overstate, so we hope the paper is accepted, on the grounds of asking so fundamental a question.

The paper correctly points out the inability of VAE to disentangle important factors even on toy data-sets. This of course has been known for awhile, e.g., Fig 4. on reference [1], from almost 2 years ago. On the back of so much hope and expectation built-up in the introduction, the solution put forward in the paper strikes us as a curious but a hardly useful toy. Slapping a large multiplicative factor on the generative error term of a VAE is not going to work for any real-life datasets. Generation without reconstruction leads to schizophrenia, as every respectable psychiatrist will  testify. Physicists have attacked this problem considerably earlier than DeepMind and their scalable and conceptual solutions have been discussed at length in references [1], section 1.6, section 4, and most of reference [2]. In short, for spatial, color, time symmetries, symmetry statistics are produced by smaller specialized nets like spatial transformers and used to augment the latent variables, to aid the decoder. As demonstrated in reference [2], Figures 3,4, this is indispensable in order to handle distortion, e.g., spatial transformation. Note that this approach is completely unsupervised. 

This of course does not handle complex factors like "type of legs" but that is a hell of an ambitious goal, and while we hope to be proven wrong, probably way beyond reach of machines and even many humans (for real-life data-sets that is!)

By complete luck, the authors may have hit upon something else of fundamental importance: the letter "beta" for their multiplicative coefficient is of course reserved in statistical physics for the inverse of the temperature. This requires the separation of generative "energy" and "entropy" and is  partially addressed in section 2.9 of reference [1]. The correct definition of "generative temperature" is not published yet, but used extensively in experiments and can be privately communicated upon request. When worked out correctly, the beta does not multiply the generative error, as in this paper, so it would be very interesting to repeat the toy experiments here with the "correct" physical model instead.

A question to the authors: creating a new metric ("disentanglement") as u do is commendable but after hundred years of PCA, ICA, etc , is there no some proxy that can be used instead? Also, could you not for example simply distort the dataset along  different factors and then look for the quality of the reconstructed images (as in the Introduction of reference [2])?

[1] https://arxiv.org/pdf/1508.06585v5.pdf
[2] https://arxiv.org/pdf/1511.02841v3.pdf
Rating: 7: Good paper, accept
Confidence: 5: The reviewer is absolutely certain that the evaluation is correct and very familiar with the relevant literature
[–]
official review by AnonReviewer2 
ICLR 2017 conference AnonReviewer2
25 Dec 2016 at 08:09ICLR 2017 conference official reviewReaders:  EveryoneShow Revisions
Review: The paper proposes beta-VAE which strengthen the KL divergence between the recognition model and the prior to limit the capacity of latent variables while sacrificing the reconstruction error. This allows the VAE model to learn more disentangled representation. 

The main concern is that the paper didn't present any quantitative result on log likelihood estimation. On the quality of generated samples, although the beta-VAE learns disentangled representation, the generated samples are not as realistic as those based on generative adversarial network, e.g., InfoGAN. Beta-VAE learns some interpretable factors of variation, but it still remains unclear why it is a better (or more efficient) representation than that of standard VAE.

In experiment, what is the criteria for cross-validation on hyperparameter \beta?

There also exists other ways to limit the capacity of the model. The simplest way is to reduce the latent variable dimension. I am wondering how the proposed beta-VAE is a better model than the VAE with reduced, or optimal latent variable dimension.
Rating: 5: Marginally below acceptance threshold
Confidence: 4: The reviewer is confident but not absolutely certain that the evaluation is correct
[–]
Response to AnonReviewer2 
Irina Higgins
09 Jan 2017 at 17:11ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: We thank the reviewer for his work, but given their review we are afraid they may have misunderstood the point of our paper and did not provide a fair assessment of our contribution.
We hope our responses below and the comments of the other reviewers may help clarify the scope of our work and its significance.

1. We do not present any quantitative result on log-likelihood estimation, because it is not the main point of our paper (as also pointed out by AnonReviewer3). This also relates to the reviewer’s comment about the quality of beta-VAE samples and how they compare to those by InfoGAN. The quality of reconstructions of beta-VAE and InfoGAN are in line with the base methods the two approaches are built upon - the VAE and GAN. Hence our reconstructions are blurrier than those of InfoGAN due to the limitations of VAEs. Saying this, InfoGAN suffers from the shortcomings of GANS, such as training instability and mode jumping. We stress again: we do not strive in this paper for good quality reconstructions; the main point of our paper is the quality of the learned representations i.e. the disentangling capability of our approach, and in this regard we clearly demonstrate that our method outperforms InfoGAN, which is the current state of the art fully unsupervised method for disentangled factor learning. In addition, we address the other main shortcoming of the InfoGAN paper (and indeed, most other unsupervised representation-learning papers): a lack of justification for the conclusion apart from weak empirical evidence, which we address by our quantitative disentanglement metric.

2. The benefits of the representations learnt by beta-VAE over those learnt by a standard VAE come from their disentangled property. The benefits of disentangled representations have been argued for by many, including Bengio (2013) and most recently in a review article on representation learning that has just come out on arxiv (https://arxiv.org/abs/1612.05299). These benefits include novel example generation through interpretable factor recombination (problems that arise from recombining entangled factors can be seen in the reduced quality of samples generated by an entangled VAE vs a disentangled beta-VAE as shown in Figure 9, which we have now added to the appendix), compression (according to the minimum description length principle), and performance in zero-shot inference, transfer and domain adaptation scenarios. We have amended our introduction to present these benefits more clearly.
Our current manuscript describes a new state of the art framework for disentangled factor learning. We leave the demonstration of benefits that arise from learning such disentangled representations to future work. We have, however, added a new section in Appendix A.5, as well as Table 2 to describe the performance of a simple classifier trying to predict the data generative factor values using different latent representations. We find that such a classifier performs significantly better using the disentangled representation of beta-VAE compared to the entangled representations of VAE.

3. We did not cross-validate the parameter \beta, as this doesn’t really make sense in an unsupervised learning setting. Indeed, as is usually the case in the literature (see InfoGAN paper for example), we train on the full dataset and assess characteristics of the representation obtained. 
As we mention in the text, in this current work we chose \beta using visual assessments of the latent traversals. Alternatively, we also mention that our disentanglement metric score could be used for this purpose. This is validated by the fact that \beta values found to be optimal through the visual assessment heuristic on the dataset of 2D shapes were also found to be within the optimal range of \beta’s as assessed with the disentanglement metric. We have now also included Appendix A.5 as further validation; in this Appendix, we use the \beta-VAE representation to directly predict the values of the ground truth factors on the 2D shapes dataset with a linear classifier. Although this does not measure disentangling directly (and hence is not used as our preferred disentangling metric), it further backs up the efficacy of \beta-VAE relative to other models in the literature.
Additionally, we always train multiple networks with random initialisations of weights for every value of \beta we tried (as can be seen in Figure 6). We did not see any strong variations in their behaviour, and hence we believe that \beta is not a particularly sensitive parameter.

4. We have trained VAEs with different latent sizes (z_size \in [2, 10]) and found that reducing z_size was linearly *reducing* the disentangling metric score. 
In particular, a VAE with the “optimal” z_size (e.g. z_size = 5 for our 2D shape dataset, which has 5 ground truth factors of variation) was not able to learn latents corresponding to the ground truth generative factors, in contrast with the beta-VAE (as shown in Fig 7 and by the disentangling metric score).
Hence beta-VAE is indeed different from a VAE with a reduced/optimal latent channel capacity.


We hope this addresses the reviewer’s concerns and welcome any further feedback.
[–]
Simple and effective 
ICLR 2017 conference AnonReviewer1
20 Dec 2016 at 00:09ICLR 2017 conference official reviewReaders:  EveryoneShow Revisions
Review: Summary
===

This paper presents Beta-VAE, an augmented Variational Auto-Encoder which
learns disentangled representations. The VAE objective is derived
as an approximate relaxation of a constrained optimization problem where
the constraint matches the latent code of the encoder to a prior.
When KKT multiplier beta on this constraint is set to 1 the result is the
original VAE objective, but when beta > 1 we obtain Beta-VAE, which simply
increases the penalty on the KL divergence term. This encourages the model to
learn a more efficient representation because the capacity of the latent
representation is more limited by beta. The distribution of the latent
representation is rewarded more when factors are independent because
the prior (an isotropic Gaussian) encourages independent factors, so the
representation should also be disentangled.

A new metric is proposed to evaluate the degree of disentanglement. Given
a setting in which some disentangled latent factors are known, many examples
are generated which differ in all of these factors except one. These examples
are encoded into the learned latent representation and a simple classifier
is used to predict which latent factor was kept constant. If the learned
representation does not disentangle the constant factor then the classifier
will more easily confuse factors and its accuracy will be lower. This
accuracy is the final number reported.

A synthetic dataset of 2D shapes with known latent factors is created to
test the proposed metric and Beta-VAE outperforms a number of baselines
(notably InfoGAN and the semi-supervised DC-IGN).

Qualitative results show that Beta-VAE learns disentangled factors
on the 3D chairs dataset, a dataset of 3D faces, and the celebA dataset
of face images. The effect of varying Beta is also evaluated using the proposed
metric and the latent factors learned on the 2D shapes dataset are explored
in detail.


Strengths
===
* Beta-VAE is simple and effective.

* The proposed metric is a novel way of testing whether ground truth factors
of variation have been identified.

* There is extensive comparison to relevant baselines.


Weaknesses
===

* Section 3 describes the proposed disentanglement metric, however I feel
I need to read the caption of the associated figure (I thank for adding
that) and Appendix 4 to understand the metric intuitively or in detail.
It would be easier to read this section if a clear intuition preceeded
a detailed description and I think more space should be devoted to this
in the paper.

* Appendix 4: Why was the bottom 50% of the resulting scores discarded?

* As indicated in pre-review comments, the disentanglement metric is similar
to a measure of correlation between latent features. Could the proposed metric
be compared to a direct measure of cross-correlation between latent factors
estimated over the 2D shapes dataset?


* The end of section 4.2 observes that high beta values result in low
disentanglement, which suggests the most efficient representation is not
disentangled. This seems to disagree with the intuition from the approach
section that more efficient representations should be disentangled. It would
be nice to see discussion of potential reasons for this disagreement.

* The writing is somewhat dense.


Overall Evaluation
===
The core idea is novel, simple and extensive tests show that it is effective.
The proposed evaluation metric is novel might come into broader use.
The main downside to the current version of this paper is the presentation,
which provides sufficient detail but could be more clear.
Rating: 7: Good paper, accept
Confidence: 4: The reviewer is confident but not absolutely certain that the evaluation is correct
[–]
Response to AnonReviewer1 
Irina Higgins
09 Jan 2017 at 17:08 (modified: 10 Jan 2017 at 08:35)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: We’d like to thank the reviewer for their thorough review and helpful suggestions.

1. We acknowledge the importance of clearly explaining our disentangled metric and we agree that our initial attempt can be improved. In order to address that, we added three new paragraphs at the beginning of Section 3 to provide an intuitive explanation for our disentanglement metric. We also rewrote the end of that section to simplify the formal presentation of the metric. We hope this provides enough background and support for why we devised it that way and what exactly it is assessing.

2. With respect to the bottom 50% question: we have added a sentence at the end of the second paragraph of Appendix A.4 to address it. The bottom 50% of the resulting scores were discarded to control for the outlier results from the few experiments that diverged during training.

3. Thank you for the suggestion. The cross-correlation fails to fully capture the properties we would look for in a disentanglement metric, as we now discuss in the new first 3 paragraphs of Section 3.
We did calculate it though (average absolute values of upper triangular terms in correlation matrix, estimated over the entire dataset), but unfortunately found that the cross-correlation does not seem to provide an appropriate measure (we’re also concerned with the large standard deviations). 
Here are the values we obtained:
PCA: 4.7e-9 (std 5e-9) // ICA: 1.15e-6 (std 9e-7) // DC-IGN: 0.15 (std 0.16) // 
InfoGAN: 0.09 (std 0.08) // VAE 0.039 (std 0.04) // beta-VAE: 0.057 (std: 0.053)
As expected, PCA and ICA produce the least correlated latents, since these approaches directly optimise for reducing cross-correlation between latents. The other approaches, including the semi-supervised DC-IGN, produce more correlated latents. However, these are nevertheless disentangled according to a human interpretation of what disentangled factors should look like.
This suggests that humanly interpretable disentangled factors of variation are still somewhat correlated in the real world, which suggests that PCA and ICA may be optimising the wrong objective for interpretable disentangling.
Hence we think that our metric provides a better test for interpretable disentangled representations, which a simple cross-correlation fails to capture.

4. We have added a sentence in the relevant paragraph in Section 4.2. If  is too high and the resulting capacity of the latent channel is lower than the number of data generative factors, then the learnt representation necessarily has to be entangled (as we are compressing more than what a factorial representation would require).

We hope this addresses the reviewer’s concerns and welcome any further feedback.

[EDIT: sorry, we initially entered wrong cross-correlation values for VAE and beta-VAE, but have now corrected them]
[–]
Small correction to response to AnonReviewer1 
Irina Higgins
10 Jan 2017 at 08:40ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: Sorry, we initially entered wrong cross-correlation values for VAE and beta-VAE in the original response, but have now corrected them.
[–]
Very interesting results, but more details and more quantitative results are needed 
ICLR 2017 conference AnonReviewer3
16 Dec 2016 at 12:17ICLR 2017 conference official reviewReaders:  EveryoneShow Revisions
Review: 
This paper proposes the beta-VAE, which is a reasonable but also straightforward generalization of the standard VAE. In particular, a weighting factor beta is added for the KL-divergence term to balance the likelihood and KL-divergence. Experimental results show that tuning this weighting factor is important for learning disentangled representations. A linear-classifier based protocol is proposed for measuring the quality of disentanglement. Impressive illustrations on manipulating latent variables are shown in the paper. 

Learning disentangled representations without supervision is an important topic. Showing the effectiveness of VAE for this task is interesting. Generalizing VAE with a weighting factor is straightforward (though reformulating VAE is also interesting), the main contribution of this paper is on the empirical side. 

The proposed protocol for measuring disentangling quality is reasonable. Establishing protocol is one important methodology contribution of this paper, but the presentation of Section 3 is still not good. Little motivation is provided at the beginning of Section 3. Figure 2 is a summary of the algorithm, which is helpful, but it still necessary to intuitively explain the motivation at the first place (e.g., what you expect if a factor is disentangled, and why the performance of a classifier can reflect such an expectation). Moreover, 1) z_diff appeared without any definition in the main text. 2) Use “decoding” for x~Sim(v,w) may make people confuse the ground truth sampling procedure w ith the trained decoder. 

The illustrative figures on traversing the disentangled factor are impressive, though image generation quality is not as good as InfoGAN (not the main point of this paper). However, 1) it will be helpful to discuss if the good disentangling quality only attribute to the beta factor and VAE framework. For example, the training data in this paper seems to be densely sampled for the visualized factors. Does the sampling density play a critical role? 2) Not too many qualitative results are provided for each experiment? Adding more figures (e.g., in appendix) to cover more factors and seeding images can strength the conclusions drawn in this paper. 3) Another detailed question related to the generalizability of the model: are the seeding image for visualizing faces from unseen subjects or subjects in the training set? (maybe I missed something here.)

Quantitative results are only presented for the synthesized 2D shape. What hinders this paper from reporting quantitative numbers on real data (e.g., the 2D and 3D face data)? One possible reason is that not all factors can be disentangled for real data, but it is still feasible to pick up some well-defined factor to measure the quantitative performance. 

Quantitative performance is only measured by the proposed protocol. Since the effectiveness of the protocol is something the paper need to justify, reporting quantitative results using simpler protocol is helpful both for demonstrating the disentangling quality and for justifying the proposed protocol (consistency with other measurement). A simple experiment is facial identity recognition and pose estimation using disentangled features on a standard test set (like in Reed et al, ICML 2014). 

In Figure 6 (left), why ICA is worse than PCA for disentanglement? Is it due to the limitation of the ICA algorithm or some other reasons? 

In Figure 6 (right), what is “factor change accuracy”? According to Appendix A.4 (which is not referred to in the main text), it is the “Disentanglement metric score”. Is that right?
If so Figure 6 (right) shows the reconstruction results for the best disentanglement metric score. Then, 1) how about random generation or traversing along a disentangled factor? 2) more importantly, how is the reconstruction/generation results when the disentanglement metric score is suboptimal. 

Overall, the results presented in this paper are very interesting, but there are many details to be clarified. Moreover, more quantitative results are also needed. I hope at least some of the above concerns can be addressed. 



Rating: 6: Marginally above acceptance threshold
Confidence: 4: The reviewer is confident but not absolutely certain that the evaluation is correct
[–]
Response to AnonReviewer3 
Irina Higgins
09 Jan 2017 at 17:10ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: We are very grateful to the reviewer for their very thorough assessment of our work and the numerous suggestions. We have made a number of changes to fully address them.

1. About our presentation of our disentanglement metric, this was a concern shared with Reviewer 3. We have now addressed this by introducing three new paragraphs at the start of Section 3 to provide an intuitive explanation of our disentanglement metric. We also rewrote the end of Section 3 to simplify the presentation.

2. Z_diff is now properly defined, sorry for this oversight.

3. We changed ‘decode’ to ‘simulate’ and ‘re-encode’ to ‘infer’ in order to avoid confusion between our disentanglement metric and the trained decoder, thank you for mentioning it.

4. We have added a sentence in Conclusions with a reference to Section A.8 in the Appendix. This section discusses the effects of data generative factor sample density on the learnt representations of beta-VAE, with empirical results.

5. We have added extra traversal plots in the appendix, with more seeding images. These include all latents that learnt to be informative when presented with 3D chairs or CelebA. 

6. About the generalizability of our model: currently all seeding images are from the same full training set. As done in this literature [e.g. InfoGAN], we did not hold out any test set. This would involve handling a domain adaptation scenario, which we are interested in tackling but decided to leave for future work.

7. On the topic of running the disentanglement metric on real datasets: we explored the possibility on the CelebA dataset, but did not report a score for two main reasons. 
First, as the reviewer suggested, many of the labels in the dataset are not those that could be reasonably expected to be disentangled, and indeed may not correspond to ground truth factors of variation. For example, ‘attractiveness’ is such a label. 
Secondly, even though there were a (small) subset of labels that could potentially disentangle well, these labels do not necessarily correspond to factors that control for large visual changes in the subject’s faces. Visually, we feel that the primary factors of variation are “background colour”, “azimuth/rotation”, “facial identity”, and “hair colour”. Of these, only hair colour was present as an available label (albeit with quite poor labelling, e.g. many subjects did not have a particular hair colour specified). 
In future work we would hope to demonstrate the efficacy of the metric on better labeled datasets, such as Caltech-UCSD Birds (http://www.vision.caltech.edu/visipedia/CUB-200.html) or Oxford Flowers (http://www.robots.ox.ac.uk/~vgg/data/flowers/).

8. The reviewer mentions using another simpler protocol to assess our disentanglement metric, this is indeed a great suggestion.
We have added a new section in Appendix A.5, with a new Table 2, to describe the performance of a simple classifier trying to predict the data generative factor values of the 2D shapes dataset using different latent representations.
We find that the results of this classification test match those of our proposed disentanglement metric, thus providing further validation for it as the reviewer was suggesting.

9. We are not entirely sure why PCA representations are more disentangled than ICA representations according to our metric. Both approaches score similarly well in terms of cross correlations between the latents, as would be expected (as mentioned to AnonReviewer1). According to our new Table 2 in Appendix A.5, the independent components learnt by ICA tend to be very well suited for object identity identification, while those learnt by PCA are more representative of the object location.

10. Thank you for finding the mention of the “factor change accuracy” term, this indeed was a typo. We updated Fig.6 to use “Disentanglement Metric Score” as expected.

11. We also now added a reference to Appendix A.4 in the main text, sorry for the oversight.

12. Finally about your points following the mention of Figure 6 (right):
We have included random samples from both disentangled and entangled models in Appendix A.9, Figure 9.
As for latent traversal visualisations for both conditions, this is actually what can already be seen in Fig. 7, panels A and B, if we understood the question correctly?

We hope this addresses the reviewer’s concerns and welcome any further feedback.
[–]
Comments to response 
ICLR 2017 conference AnonReviewer3
21 Jan 2017 at 10:47ICLR 2017 conference official commentReaders:  EveryoneShow Revisions
Comment: Thank you for the authors' detailed reply. The descriptions of the evaluation metric are much better. 

The discussion in A.8 is very helpful. In the meanwhile, learning disentangled features with beta-VAE on relatively discontinuous data also seems to be difficult. 

For Table 2, the results are fine, but also a bit mixed (compared to PCA). Moreover, in order to show the degree of disentanglement, it is also important to show the classification rate using features mismatched with the tasks (e.g., id recognition using rotation features). I understand it may be cumbersome to report all possible combinations, but focusing on one or two recognition tasks (e.g., the id recognition) is definitely doable. 

In general, I recommend acceptance for this paper, because 1) disentangling factors of variations in an unsupervised way is important; 2) the visualization results are interesting. In the meanwhile, the quantitive results and practical significance are not so strong as expected (I am not saying they are too weak). 
[–]
public comment by 
[Deleted]
ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
[–]
Questions about the disentanglement metric 
ICLR 2017 conference AnonReviewer1
04 Dec 2016 at 15:03 (modified: 04 Dec 2016 at 15:03)ICLR 2017 conference questionReaders:  EveryoneShow Revisions
Question: * I'm having trouble understanding the proposed disentanglement metric. Here's my attempt at summarizing the metric

1. Choose some factor k.
2. Sample two sets of latent representations, A and B, one with perturbed v_k.
3. Decode the re-encode each latent representation.
4. Use the difference between re-encoded latent representations to predict k and measure the accuracy of this predictor.

If the latent representation v_k is disentangled then step 4 will be easy, otherwise it will be hard. Thus, the performance of the predictor measures something like disentanglement.

Is that right? Could section 3 be clarified in the paper?

* Supposing my understanding of the metric is accurate, if v_k is correlated with other latent factors then it's impossible to distinguish between v_k and the correlated factors using a simple classifier. This makes the prediction task in step 4 hard. Otherwise, when v_k is not correlated with other factors, it's possible that z_diff can distinguish between v_ks, so step 4 could be easy. If this is the right idea, then is the proposed metric simply saying that a representation is disentangled when its components are not correlated with each other?
[–]
Disambiguating the disentanglement metric 
Irina Higgins
08 Dec 2016 at 09:35 (modified: 08 Dec 2016 at 09:35)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: The intuition is almost correct. In the terms suggested by the reviewer, the correct intuition would be:


1. Choose some factor k.
2. For a batch of N samples:
a. Sample two sets of latent representations, A and B, where v_k^A = v_k^B (and all other v_i may change).
	b. Decode then re-encode each latent representation.
	c. Calculate the difference z_diff between re-encoded latent representations 
3. Use the average z_diff over N samples to predict k and measure the accuracy of this predictor.


We have modified Section 3 to add an intuitive clarification.


Your intuition is correct - the metric does say that a representation is disentangled when its components are not correlated, whereby p(z) = ∏_i ∫_x p(z_i | x) . The components may be conditionally dependent given a particular image though: p(z_x) != ∏_i p(z_i | x). The metric can deal with such conditional dependencies due to its batch averaging of z_diff.


Also the model can only learn to disentangle factors that are statistically independent in the data. For example, if in CelebA the majority of young people are female and the majority of older people are male, the best one can hope for is for the model to disentangle out a joint factor of age/gender (as seen in Figure 4). 
[–]
About the assumption and measurement 
ICLR 2017 conference AnonReviewer3
02 Dec 2016 at 21:54 (modified: 02 Dec 2016 at 21:54)ICLR 2017 conference questionReaders:  EveryoneShow Revisions
Question: (v,w) categorizes the latent representations into disentangled dimensions and non-disentangled dimensions. Is it possible to consider group-wise disentangling? This is more practical for complex factors, such as human identity.

The measurement proposed in Section 3 is intuitively to measure if a certain dimension (or certain dimensions) has lower variance when a certain factor is fixed. Compared to more basic measurements, such as variance, what are the benefits of using a linear classifier? 

The presentation of Section 3 is confusing. Is it possible to make the intuition clearer in the main text?

In Section 4, was the measurement in Section 3 maximized to determine beta? (I have not read all details in Section 4, so I might just miss this)


[–]
Clarifying section 3 
Irina Higgins
08 Dec 2016 at 09:37 (modified: 08 Dec 2016 at 09:37)ICLR 2017 conference public commentReaders:  EveryoneShow Revisions
Comment: What do you mean by ‘group-wise disentangling’? Do you mean conditionally dependent factors (i.e. given identity X, factors such as face shape, eye colour, hair style etc can be explained away)? If this is the case, we believe that identity would be a ‘complex’ factor consisting of a particular combination of more ‘simple’ factors of face shape, eye colour etc. Learning such ‘complex’ factors would require learning hierarchical latent representations, which is something we do not address in the current work but leave for future efforts. The present paper only attempts to learn a set of ‘simple’ data generative factors that lie at the bottom of such potential hierarchy.


Our model can only learn to disentangle factors that are statistically independent in the data. If in the data certain features are strongly correlated (e.g. full lips often co-occur with wider noses due to the two features being encoded in the same genes), then our model will learn a factor that encodes both of these features, resulting in ‘group-wise disentangling’ of a sort. An example of such coding can be found in CelebA, where it appears that the majority of young people are female and the majority of older people are male. Hence our model learns to disentangle out a ‘group-wise’ factor of age/gender (as seen in Figure 4). 


Linear classifier helps our disentanglement metric have a single interpretable score that we can use to better compare different models in a way that reflects disentanglement. Calculating the average variance per factor per X batches would be a worse measure, because we are interested in factor k having lower variance on average compared to other factors i != k. In absolute terms such average variance is not informative with regards to the level of disentanglement achieved. We believe that using a linear classifier is one of the easier ways to quantify whether factor k has lower average variance.


We have added Figure 2 in Section 3 to add an intuitive clarification of our proposed metric.


We did not explicitly try to maximise the disentanglement metric to determine beta, because this metric works in a semi-supervised case and we wanted to demonstrate that beta-VAE works in a completely unsupervised setting. We did show, however, that the best values of betas chosen through visual inspection heuristics match well the optimal betas found through calculating the disentanglement metric (see Fig. 5 right)
About OpenReview
Contact
FAQ
Hosting a Venue
Sponsors
Terms of Use / Privacy Policy
All Venues
Donate
News
OpenReview is a long-term project to advance science through improved peer review with legal nonprofit status. We gratefully acknowledge the support of the OpenReview Sponsors. © 2026 OpenReview