# Note when execution began
startTime = Dates.now()

using Flux
using Flux: @epochs
using JSON
using BSON: @save
using HTTP
using StatsBase
using MicroLogging
using MLDataUtils
using Distributions

# Set up logging to file
dfmt = dateformat"yyyy-mm-dd-HHMMSS"
logFile = splitext(basename(PROGRAM_FILE))[1] * "-$(Dates.format(startTime, dfmt)).log"
logger = MicroLogging.InteractiveLogger(open(logFile, "w"))
MicroLogging.configure_logging(logger)

# Download complete dataset
const BASE_URL = "http://www.trumptwitterarchive.com/data/realdonaldtrump"
const STOP_CHAR = '¶'
const UNKNOWN_CHAR = '�'
const MAX_CHARS = 280

"""getJSONTweets(year)

Download tweets from the Trump Twitter Archive for the given year.

Returns an Array of Dict{String,Any}."""
function getJSONTweets(year::Integer)
    url = BASE_URL * "/$year.json"
    @info "Downloading" url
    r = HTTP.request("GET", url)
    # TODO: check r.status
    JSON.parse(transcode(String, r.body))
end

"""makeAlphabet(tweets; cutoff)

Scans through the given tweets and generates an alphabet of all used characters.
Drops characters that have a usage rate per tweet of less than `cutoff`.  For
instance, if 100 tweets are given and `cutoff = 0.1`, then characters appearing
in fewer than 10% of the tweets will be ignored.

Returns an Array of Chars."""
function makeAlphabet(tweetTexts::AbstractVector{String}; cutoff::Real = 1e-3)
    @assert cutoff > 0 && cutoff <= 1 "cutoff must be in the range (0,1], $cutoff given"

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

"""padAndTrim(m, padVec, len)

Pads OneHotMatrix `m` to the right out to `len` columns, filling with `padVec`
as needed.  If the number of columns in `m` is greater than `len`, trims `m` to
`len` columns.  Also, forces the last column of `m` to be `padVec` no matter
its original value.

Returns a new `OneHotMatrix` with the same number of rows as `m`, but with
exactly `len` columns."""
function padAndTrim(m::Flux.OneHotMatrix, padVec::Flux.OneHotVector, len::Integer)
    d2 = min(size(m, 2), len)
    m_ = Flux.OneHotMatrix(m.height, [m[:, 1:d2].data; fill(padVec, max(len - d2, 0))])
    m_.data[d2] = padVec
    m_
end

# Collect the tweets into a single array of strings
tweets = Iterators.flatten(getJSONTweets(y) for y in 2009:Dates.year(Dates.now()))
@info "# Processing tweets"
tweetTexts = collect(normalize_string(t["text"], :NFKC) for t in tweets)

@info "# Making alphabet"
alphabet = makeAlphabet(tweetTexts)
nChars = length(alphabet)

stopVec = Flux.onehot(STOP_CHAR, alphabet)
makeOneHots(texts::Vector{String}, alphabet::Vector{Char}, batchSize::Integer) = Iterators.partition((padAndTrim(Flux.onehotbatch(text, alphabet, UNKNOWN_CHAR), stopVec, MAX_CHARS) for text in texts), batchSize)
# Predictions are just the next character in line,
# except that the last character is always STOP_CHAR.
makePredicted(texts::Vector{String}, alphabet::Vector{Char}, batchSize::Integer) = Iterators.partition((padAndTrim(Flux.onehotbatch(text[chr2ind(text,2):end] * STOP_CHAR, alphabet, UNKNOWN_CHAR), stopVec, MAX_CHARS) for text in texts), batchSize)

@info "# Converting to one-hots"
batchSize = 64
Xs = collect(makeOneHots(tweetTexts, alphabet, batchSize))
Ys = collect(makePredicted(tweetTexts, alphabet, batchSize))

#@info "Creating predictors and prediction targets"
#Xs = Flux.chunk(tweetOneHots, nBatch)
# Predict the very next character
#Ys = Flux.chunk(predictedOneHots, nBatch)

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

function lossCallback()
    idx = rand(1:length(Xs))
    println("$(Dates.now()): $(loss(Xs[idx], Ys[idx]))")
end


function twsample(p::AbstractVector; temperature::Real = 1.)
    p_ = p ./ sum(p)
    p_ = log.(p_) ./ temperature
    p_ = exp.(p_)
    p_ = p_ ./ sum(p_)
    m = Distributions.Multinomial(1, p_)
    Flux.argmax(rand(m))
end

function sampleTweet(m, alphabet, len; temp = 1)
    # TODO: implement temperature
    Flux.reset!(m)
    buf = IOBuffer()
    c = rand(alphabet)
    for i in 1:len
        write(buf, c)
        if c == STOP_CHAR
            break
        end
        c = alphabet[twsample(m(Flux.onehot(c, alphabet)).data, temperature = temp)]
    end
    String(take!(buf))
end

function sampleCallback()
    println("$(Dates.now()): $(sampleTweet(model, alphabet, 280; temp = 0.9))")
end

nEpochs = 10
for epoch in 1:nEpochs
    @info "Training" progress=epoch/nEpochs
    data = MLDataUtils.shuffleobs((Xs, Ys))
    Flux.train!(loss, data, optimizer,
        cb=[Flux.throttle(lossCallback, 30), Flux.throttle(sampleCallback, 120)])
end

@save "char-gru.bson" model
