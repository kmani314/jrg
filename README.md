# JRG
Japanese rap lyric generator using transformers

This is an RNN variant of the model which is currently trained on a subset of japanese wikipedia and sentences from tatoeba. Unfortunately, Japanese rap lyrics and quite hard to come by, so training a model only on rap lyrics may be difficult.

Here are some cherrypicked examples, which continue to surprise me. The prompt is 「彼は」(He...):

> 彼は酔っていた。家と呼んだ。
> He was drunk. He called it home.

> きっと時々沈黙がある。
> Surely there is silence from time to time.

> 彼はそれを許していますか。 
> Does he forgive it?

> その静かな空はないんです。
> There is no such peaceful sky.

> 光ったらなんて成功しているところを必要でできます
> (partially ungrammatical) If it shines, you are able to need to be successful

It's also capable of being quite philosophical. The prompt is 「命というのは」("Life" is...):

> 命というのはひどく無いのは難しい。
> The difficult things in life are those that aren't horrible.

> 命というのは寒い。
> Life is chilly.

> 命というのは満足しないと難しい。
> Life is difficult if you're not satisfied.

> 命というのは、適切でした。
> Life was adequate.

> 命というのはよく思わない。
> I don't really think about life.

「死というのは」("Death is..."):

> 死というのは間違いのですか。 
> Is death is mistake?

> 死というのは困難だ。
> Death is hardship.

> 死というのは美味しかった。
> Death was delicious.
