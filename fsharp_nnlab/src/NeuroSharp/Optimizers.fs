namespace NeuroSharp

open NeuroSharp.Math

type ParameterGradients =
    {
        WeightGradients: float[,]
        BiasGradients: float[]
    }

type ParameterState =
    {
        Weights: float[,]
        Biases: float[]
    }

type Optimizer =
    abstract member InitializeLayer : int -> ParameterState -> unit
    abstract member Update : int -> ParameterState -> ParameterGradients -> unit

type SgdOptimizer(learningRate: float) =
    interface Optimizer with
        member _.InitializeLayer(_, _) = ()

        member _.Update(_, state, grads) =
            for r in 0 .. rows state.Weights - 1 do
                for c in 0 .. cols state.Weights - 1 do
                    state.Weights[r, c] <- state.Weights[r, c] - learningRate * grads.WeightGradients[r, c]
            for i in 0 .. state.Biases.Length - 1 do
                state.Biases[i] <- state.Biases[i] - learningRate * grads.BiasGradients[i]

type MomentumOptimizer(learningRate: float, momentum: float) =
    let weightVelocity = System.Collections.Generic.Dictionary<int, float[,]>()
    let biasVelocity = System.Collections.Generic.Dictionary<int, float[]>()

    interface Optimizer with
        member _.InitializeLayer(index, state) =
            weightVelocity[index] <- zeros (rows state.Weights) (cols state.Weights)
            biasVelocity[index] <- zerosVector state.Biases.Length

        member _.Update(index, state, grads) =
            let vw = weightVelocity[index]
            let vb = biasVelocity[index]
            for r in 0 .. rows state.Weights - 1 do
                for c in 0 .. cols state.Weights - 1 do
                    vw[r, c] <- momentum * vw[r, c] - learningRate * grads.WeightGradients[r, c]
                    state.Weights[r, c] <- state.Weights[r, c] + vw[r, c]
            for i in 0 .. state.Biases.Length - 1 do
                vb[i] <- momentum * vb[i] - learningRate * grads.BiasGradients[i]
                state.Biases[i] <- state.Biases[i] + vb[i]

type GradientClippingOptimizer(inner: Optimizer, maxNorm: float) =
    interface Optimizer with
        member _.InitializeLayer(index, state) =
            inner.InitializeLayer(index, state)

        member _.Update(index, state, grads) =
            inner.Update(index, state, {
                WeightGradients = clipByNorm maxNorm grads.WeightGradients
                BiasGradients = clipByNormVector maxNorm grads.BiasGradients
            })

[<RequireQualifiedAccess>]
module Optimizers =
    let sgd lr = SgdOptimizer(lr) :> Optimizer
    let momentum lr beta = MomentumOptimizer(lr, beta) :> Optimizer
    let clip maxNorm (inner: Optimizer) = GradientClippingOptimizer(inner, maxNorm) :> Optimizer
