using Flux
using JSON
using HTTP
using StatsBase

# Download complete dataset
const BASE_URL = "http://www.trumptwitterarchive.com/data/realdonaldtrump"
const STOP_CHAR = '¶'
const UNKNOWN_CHAR = '�'

function getJSONTweets(year::Integer)
    url = BASE_URL * "/$year.json"
    r = HTTP.request("GET", url)
    # TODO: check r.status
    JSON.parse(transcode(String, r.body))
end

# Collect the tweets into a single arraw of strings
tweets = Iterators.flatten(getJSONTweets(y) for y in 2009:2018)
tweetTexts = collect(normalize_string(t["text"], :NFKC) for t in tweets)
fullText = join(tweetTexts, STOP_CHAR)

# Get the frequencies of character use and make a sensible cut
freqs = Flux.frequencies(text)
alphabetCutoff = 1e-3
alphabet = [pair[1] for pair in freqs if pair[2]/length(tweetTexts) > alphabetCutoff]
if !in(STOP_CHAR, alphabet)
    append!(alphabet, stopChar)
end
if !in(UNKNOWN_CHAR, alphabet)
    append!(alphabet, UNKNOWN_CHAR)
end

# Convert into onehots
textData = [Flux.onehot(ch, alphabet, UNKNOWN_CHAR) for text in tweetTexts for ch in text]

N = length(alphabet)
seqLength = 100
nBatch = 64
stopVec = Flux.onehot(STOP_CHAR, alphabet)

Xs = collect(Iterators.partition(Flux.batchseq(Flux.chunk(textData, nBatch), stopVec), seqLength))
# Predict the very next character
Ys = collect(Iterators.partition(Flux.batchseq(Flux.chunk(textData[2:end], nBatch), stopVec), seqLength))

model = Flux.Chain(
    Flux.GRU(N, 128),
    Flux.GRU(128, 128),
    Flux.Dense(128, N),
    Flux.softmax)

function loss(xs, ys)
    l = sum(Flux.crossentropy.(model.(xs), ys))
    Flux.truncate!(model)
    return l
end

optimizer = Flux.ADAM(Flux.params(model), 0.01)

lossCallback = () -> @show loss(Xs[5], Ys[5]) # random number determined by roll of fair die

function sampleTweet(m, alphabet, len; temp = 1)
    # TODO: implement temperature
    Flux.reset!(m)
    buf = IOBuffer()
    c = rand(alphabet)
    for i in 1:len
        write(buf, c)
        c = StatsBase.wsample(alphabet, m(Flux.onehot(c, alphabet)).data)
    end
    String(take!(buf))
end

sampleCallback = () -> @show sampleTweet(model, alphabet, 140; temp = 0.9)

Flux.train!(loss, zip(Xs, Ys), optimizer,
    cb=[Flux.throttle(lossCallback, 30)])
