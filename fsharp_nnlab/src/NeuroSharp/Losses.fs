namespace NeuroSharp

open NeuroSharp.Math

type Loss =
    {
        Name: string
        Value: float[,] -> float[,] -> float
        Gradient: float[,] -> float[,] -> float[,]
    }

[<RequireQualifiedAccess>]
module Losses =
    let mse =
        {
            Name = "MSE"
            Value = fun yTrue yPred ->
                let diff = subtract yPred yTrue
                meanSquaredNormMatrix diff

            Gradient = fun yTrue yPred ->
                let scale = 2.0 / float (rows yTrue * cols yTrue)
                subtract yPred yTrue |> scalarMul scale
        }

    let binaryCrossEntropy =
        {
            Name = "BinaryCrossEntropy"
            Value = fun yTrue yPred ->
                let eps = 1e-9
                let mutable acc = 0.0
                for r in 0 .. rows yTrue - 1 do
                    for c in 0 .. cols yTrue - 1 do
                        let p = clamp eps (1.0 - eps) yPred[r, c]
                        let y = yTrue[r, c]
                        acc <- acc - (y * log p + (1.0 - y) * log (1.0 - p))
                acc / float (rows yTrue * cols yTrue)

            Gradient = fun yTrue yPred ->
                let eps = 1e-9
                createMatrix (rows yTrue) (cols yTrue) (fun r c ->
                    let p = clamp eps (1.0 - eps) yPred[r, c]
                    let y = yTrue[r, c]
                    (p - y) / ((p * (1.0 - p)) * float (rows yTrue * cols yTrue)))
        }

    let categoricalCrossEntropy =
        {
            Name = "CategoricalCrossEntropy"
            Value = fun yTrue logits ->
                let probs = softmax logits
                let eps = 1e-9
                let mutable acc = 0.0
                for r in 0 .. rows yTrue - 1 do
                    for c in 0 .. cols yTrue - 1 do
                        if yTrue[r, c] > 0.0 then
                            acc <- acc - log (clamp eps 1.0 probs[r, c])
                acc / float (rows yTrue)

            Gradient = fun yTrue logits ->
                let probs = softmax logits
                subtract probs yTrue |> scalarMul (1.0 / float (rows yTrue))
        }

module Metrics =
    let accuracy (yTrue: float[,]) (logits: float[,]) =
        let probs = Math.softmax logits
        let mutable correct = 0
        for r in 0 .. Math.rows yTrue - 1 do
            let target = Math.argmaxRow yTrue r
            let pred = Math.argmaxRow probs r
            if target = pred then correct <- correct + 1
        float correct / float (Math.rows yTrue)
