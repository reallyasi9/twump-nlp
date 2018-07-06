# Note when execution began
startTime = Dates.now()

#using CuArrays
using Flux
using Flux: @epochs
using JSON
using BSON: @save
using HTTP
using StatsBase
using MicroLogging

# Set up logging to file
dfmt = dateformat"yyyy-mm-dd-HHMMSS"
logFile = open(splitext(basename(PROGRAM_FILE))[1] * "-$(Dates.format(startTime, dfmt)).log", "w")
logger = MicroLogging.SimpleLogger(logFile)
MicroLogging.global_logger(logger)

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

"""makeAlphabet(text; cutoff)

Scans through the given text and generates an alphabet of all used characters.
Drops characters that have a usage rate per character of less than `cutoff`.  For
instance, if  `cutoff = 0.001`, then characters appearing less than 0.1% of the
time will be ignored.

Returns an Array of Chars."""
function makeAlphabet(text::AbstractString; cutoff::Real = 0)
    @assert cutoff >= 0 && cutoff <= 1 "cutoff must be in the range [0,1], $cutoff given"

    fullText = join(tweetTexts)
    freqs = Flux.frequencies(text)
    alphabet = [pair[1] for pair in freqs if pair[2]/length(text) > cutoff]
    if !in(STOP_CHAR, alphabet)
        append!(alphabet, STOP_CHAR)
    end
    if !in(UNKNOWN_CHAR, alphabet)
        append!(alphabet, UNKNOWN_CHAR)
    end
    return alphabet
end

# Collect the tweets into a single array of strings
tweets = Iterators.flatten(getJSONTweets(y) for y in 2009:Dates.year(Dates.now()))
@info "# Processing tweets"
tweetTexts = collect(normalize_string(t["text"], :NFKC) for t in tweets)
tweetText = join(tweetTexts, STOP_CHAR)

@info "# Making alphabet"
alphabet = makeAlphabet(tweetText, cutoff=1e-5)
nChars = length(alphabet)

stopVec = Flux.onehot(STOP_CHAR, alphabet)
function makeOneHots(text::AbstractString, alphabet::Vector{Char};
        batches::Integer = 64, sequenceLength::Integer = MAX_CHARS,
        start::Integer = 1, stopChar::Char = STOP_CHAR,
        padChar::Char = UNKNOWN_CHAR)
    stopText = string(stopChar)
    stopVec = Flux.onehot(stopChar, alphabet)
    t = text[chr2ind(text, start):end] * stopText^start
    x = [Flux.onehot(ch, alphabet, padChar) for ch in t]
    x = Flux.chunk(x, batches)
    x = Flux.batchseq(x, stopVec)
    x = Iterators.partition(x, sequenceLength)
    collect(x)
end

@info "# Converting to one-hots"
nBatches = 64
Xs = makeOneHots(tweetText, alphabet, batches = nBatches)
Ys = makeOneHots(tweetText, alphabet, batches = nBatches, start = 2)

model = Flux.Chain(
    Flux.GRU(nChars, 512),
    Flux.GRU(512, 256),
    Flux.GRU(256, 128),
    Flux.Dense(128, nChars),
    Flux.softmax)

#model = gpu(model)

function loss(xs, ys)
    #xs = gpu.(xs)
    #ys = gpu.(ys)
    l = sum(Flux.crossentropy.(model.(xs), ys))
    Flux.truncate!(model)
    return l
end

optimizer = Flux.ADAM(Flux.params(model), 0.01)

function lossCallback()
    idx = rand(1:length(Xs))
    #tx, ty = (gpu.(Xs[idx]), gpu.(Ys[idx]))
    #tx, ty = (Xs[idx], Ys[idx])
    @info "$(Dates.now()) Approximate loss" idx loss(Xs[idx], Ys[idx])
end


function twsample(p::AbstractVector; temperature::Real = 1.)
    if temperature > 0
        p_ = log.(p) ./ temperature
        p_ = exp.(p_)
        p_ = p_ ./ sum(p_)
        StatsBase.sample(StatsBase.weights(p_))
    else
        Flux.argmax(p)
    end
end

function sampleTweet(m, start::AbstractString, alphabet::Vector{Char}, len::Integer; temp::Real = 1)
    #m = cpu(m)
    Flux.reset!(m)
    buf = IOBuffer()

    c2 = ' '
    for c in start
        write(buf, c)
        v = Flux.onehot(c, alphabet, UNKNOWN_CHAR)
        c2 = alphabet[Flux.argmax(m(v).data)]
    end

    for i in 1:(len - length(start))
        write(buf, c2)
        if c2 == STOP_CHAR
            break
        end
        v = Flux.onehot(c2, alphabet, UNKNOWN_CHAR)
        c2 = alphabet[twsample(m(v).data, temperature = temp)]
    end

    String(take!(buf))
end

function sampleCallback()
    start = split(rand(tweetTexts))[1] * " "
    for temp in 0.:0.25:1.25
        tweet = sampleTweet(model, start, alphabet, MAX_CHARS; temp = temp)
        @info "$(Dates.now()) Sample generated tweet" start temp tweet
    end
    nothing
end

nEpochs = 100
for epoch in 1:nEpochs
    @info "Training" progress=epoch/nEpochs
    p = randperm(length(Xs))
    Flux.train!(loss, Iterators.zip(Xs[p], Ys[p]), optimizer,
        cb=[Flux.throttle(lossCallback, 30), Flux.throttle(sampleCallback, 60)])
    outFile = "char-gru_epoch$(epoch).bson"
    @info "Batches complete, saving model" outFile
    @save outFile model
end

close(logFile)
