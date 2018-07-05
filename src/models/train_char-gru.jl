using Flux
using Flux: @epochs
using JSON
using BSON: @save
using HTTP
using StatsBase
using MicroLogging

# Download complete dataset
const BASE_URL = "http://www.trumptwitterarchive.com/data/realdonaldtrump"
const STOP_CHAR = '¶'
const UNKNOWN_CHAR = '�'
const MAX_CHARS = 280

function getJSONTweets(year::Integer)
    url = BASE_URL * "/$year.json"
    @info "Downloading" url
    r = HTTP.request("GET", url)
    # TODO: check r.status
    JSON.parse(transcode(String, r.body))
end

# Collect the tweets into a single array of strings
tweets = Iterators.flatten(getJSONTweets(y) for y in 2009:Dates.year(Dates.now()))
@info "# Processing tweets"
tweetTexts = collect(normalize_string(t["text"], :NFKC) for t in tweets)

function makeAlphabet(tweetTexts::Vector{String}; cutoff = 1e-3)
    fullText = join(tweetTexts)
    freqs = Flux.frequencies(fullText)
    alphabet = [pair[1] for pair in freqs if pair[2]/length(tweetTexts) > cutoff]
    if !in(STOP_CHAR, alphabet)
        append!(alphabet, STOP_CHAR)
    end
    if !in(UNKNOWN_CHAR, alphabet)
        append!(alphabet, UNKNOWN_CHAR)
    end
    return alphabet
end

@info "# Making alphabet"
alphabet = makeAlphabet(tweetTexts)
nChars = length(alphabet)

function padAndTrim(m::Flux.OneHotMatrix, padVec::Flux.OneHotVector, len::Integer)
    d2 = min(size(m, 2), len)
    m_ = Flux.OneHotMatrix(m.height, [m[:, 1:d2].data; fill(padVec, max(len - d2, 0))])
    m_.data[d2] = padVec
    m_
end

stopVec = Flux.onehot(STOP_CHAR, alphabet)
makeOneHots(texts::Vector{String}, alphabet::Vector{Char}) = [padAndTrim(Flux.onehotbatch(text, alphabet, UNKNOWN_CHAR), stopVec, MAX_CHARS) for text in texts]
# Predictions are just the next character in line,
# except that the last character is always STOP_CHAR.
makePredicted(texts::Vector{String}, alphabet::Vector{Char}) = [padAndTrim(Flux.onehotbatch(text[chr2ind(text,2):end] * STOP_CHAR, alphabet, UNKNOWN_CHAR), stopVec, MAX_CHARS) for text in texts]

@info "# Converting to one-hots"
tweetOneHots = makeOneHots(tweetTexts, alphabet)
predictedOneHots = makePredicted(tweetTexts, alphabet)

nBatch = 64

@info "Creating predictors and prediction targets"
Xs = Flux.chunk(tweetOneHots, nBatch)
# Predict the very next character
Ys = Flux.chunk(predictedOneHots, nBatch)

model = Flux.Chain(
    Flux.GRU(nChars, 128),
    Flux.GRU(128, 128),
    Flux.Dense(128, nChars),
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

function sampleCallback()
    println("$(Dates.now()): $(sampleTweet(model, alphabet, 140; temp = 0.9))")
end

@epochs 20 Flux.train!(loss, zip(Xs, Ys), optimizer,
    cb=[Flux.throttle(lossCallback, 30), Flux.throttle(sampleCallback, 120)])

@save "char-gru.bson" model
