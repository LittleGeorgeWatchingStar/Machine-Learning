# Load X and y variable
using JLD
using PyCall
X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")
n = size(X,1)



println(size(X)[1])

#Xtrain = X[1:200,:]
#ytrain = y[1:200,:]

Xtrain = X[201:end,:]
ytrain = y[201:end,:]

XValid = X[1:200,:]
yvalid = y[1:200,:]

maxDepth = 10
include("decisionTree_infoGain.jl")
Xtest = load("citiesSmall.jld","Xtest")
ytest = load("citiesSmall.jld","ytest")

trainError = []
testError = []

for depth in 1:maxDepth
    model = decisionTree_infoGain(Xtrain,ytrain,depth)
    # Evaluate the trianing error
    yhat = model.predict(Xtrain)
    push!(trainError, sum(yhat .!= ytrain)/n)

    # Evaluate the test error
    t = size(XValid, 1)
    yhat = model.predict(XValid)
    push!(testError, sum(yhat .!= yvalid)/t)
end

@pyimport numpy
@pyimport pylab
pylab.plot(1:10, trainError; color="red", linewidth = 2.0, linestyle="--")
pylab.plot(1:10, testError; color="blue", linewidth = 2.0, linestyle="-")
pylab.show()





# Evaluate the test error

#t = size(Xtest,1)
#yhat = model.predict(Xtest)
#testError = sum(yhat .!= ytest)/t
#@printf("Test error with depth-%d decision tree: %.3f\n",depth,testError)
