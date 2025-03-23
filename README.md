# Fetch Take Home
Hello Fetch! I had a lot of fun with this take home; so much so that I actually went ahead and implemented and trained multiple models. I've been following various sources on building GPT models from scratch so my approach was to first train a base encoder on the task of next token prediction, then initialize a sentence model with that and fine tune it on a topic classification task, but even small models were close to hundreds of megabytes and I wasn't sure if it was a great idea to make anyone cloning this repo deal with that. Instead wrote a script to train a sentence encoder topics and produce embeddings. 

## Task 1: Sentence Transformer Implementation
The Sentence Transformer class is in `llm_tools/modules/models.py`. As noted above, my approach was to first build up base encoder and positional embedding layers. Since the output of the transformer is (number of tokens, embedding dimensions), I added mean pooling over the input tokens to create a sentence vector, taking care to not average over padding tokens that aren't being attended to. For the backbone, I decided to use a Pre-LayerNorm structure as opposed to Post-LayerNorm since it's generally been shown to produce more stable gradients.

The `train_sentence_model.py` script trains the sentence model from freshly initialized weights, but ideally I'd train a base encoder model that I could use on multiple downstream NLP tasks, then fine-tune a Sentence Encoder.

## Task 2: Multi-Task Learning
To implement the MTL model I expanded the Sentence Transformer with additional heads, one for each task. The forward call accepts a `task` parameter that's used to decide which head to use. Since this is mostly just a proof-of-concept I simply used shallow linear layers for each head but in a normal setting the heads would likely need to be more complicated.

## Task 3: Training Considerations
#### If the entire network should be frozen
I'm not sure if I might be misunderstanding, but if the entire network is frozen then the model cannot be trained, so I'm confused about what there is to consider. A scenario I can think of for this situation is if I wanted to use this model as the base for another model and I wanted to avoid any kind of catastrophic forgetting in this network when training on new tasks.

#### If only the transformer backbone should be frozen
Freezing the backbone can useful when resources are limited and the tasks we're training for are similar in nature. We'd have to make sure the tasks of the MTL model weren't too different from each other as a frozen backbone would mean a model that likely wouldn't generalize well to some or all of the tasks. With a frozen backbone we may have to increase the complexity of the task heads and add layers to them for the model to capture the patterns it needs to accomplish the task.

#### If only one of the task-specific heads (either for Task A or Task B) should be frozen
We might want to freeze one of the heads if performance on it is already good and we are introducing a new task. It would save time and resources if we didn't have to train the original task concurrently but we'd have to take care that the base model's weights didn't change so much that performance on the frozen task didn't become unacceptably worse (catastrophic forgetting). We can guard against this by ensuring the tasks are all similar to each other, or by preventing large changes in weights by limiting the magnitude of gradients.

#### Benefits of Transfer Learning
When we are considering transfer learning it's because we could benefit from large, complex models that are already trained on tasks adjacent to one we're working on. We may lack the data, time, or compute resources to train these models from scratch so we can use pre-trained ones to bridge the gap.

When choosing a pre-trained model we want to select one trained on a task as similar to ours as possible, but we also need to take note of a few other things. If our model is going to run in production, then we might be limited by the size and complexity of the pre-trained model based on latency or compute resource restrictions.

In general, when fine-tuning a pre-trained model we want to unfreeze as many layers as possible so that the network can tune to our tasks as best as possible. This means longer training time and/or more compute resources, however, so we can start by freezing all but the last layer/few layers and unfreeze layers from end to start as needed until the model performs as well as we need it to.

## Task 4: Training Loop Implementation
Though the instructions said I didn't need to actually train an MTL model I was really interested and went ahead with real data. I opted to use a small version of the AG News set since it has topic labels, and the IMDB movie set for sentiment classification since they were easily accessible and I felt that the two tasks were at least moderately similar. 

#### Handling the Data
The IMDB dataset was much larger than the AG News set I had and I knew that if I used the whole IMDB set then it was likely that the base encoder layers would just become largely tuned to it alone so I cut it down to be equal in size to the AG dataset.

#### Handling the Forward Pass
For the forward pass I had to pass a `task` parameter to the model so that I could switch  heads based on that. Some models infer the task based on context from the input but that would have been outside the scope of this project.

#### Handling the Backward Pass
I only know of two ways to handle the backward pass: alternate between tasks, or learn tasks simultaneously by running both tasks through the model back-to-back and add their losses together before calling `backward()`. Between the two, the first is easier to understand and debug so I chose that. I had to find some way to iterate through the dataloaders of both datasets in a round-robin fashion which I ended up implementing in the `roundrobin_iters` function. Because the lexicon between samples of news and movie review datasets was likely to be very different I knew there would be tokens in one dataset that would have never appeared in the other so this wouldn't be a situation where I would able to freeze layers unless I had a pre-trained base model that was trained on a large, general dataset.

#### Metrics
With more time, I would have liked to track the losses between the tasks both for train and validation better than what I currently have. I also would have liked to add confusion matrices. During one instance of training I ran into a situation where it appeared that gradients might be vanishing so I added gradient tracking. Of course, after I added this it never happened again but it was still interesting to see.

## Other Decisions
I'm a huge fan of MyPy and type-hints so I try to use them at all times. I also love using [black](https://github.com/psf/black) and [isort](https://pycqa.github.io/isort/) for code formatting. I love Pytest for testing and am a big fan of writing tests for the majority of my code.