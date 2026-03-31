namespace NeuroSharp

open NeuroSharp.Math

type DenseCache =
    {
        Input: float[,]
        LinearOutput: float[,]
        ActivatedOutput: float[,]
    }

type DenseLayer(inputSize: int, outputSize: int, activation: Activation) =
    let scale = sqrt (2.0 / float (inputSize + outputSize))

    let weights =
        createMatrix inputSize outputSize (fun _ _ ->
            (random.NextDouble() * 2.0 - 1.0) * scale)

    let biases = Array.zeroCreate<float> outputSize
    let mutable cache = None: DenseCache option

    member _.InputSize = inputSize
    member _.OutputSize = outputSize
    member _.Activation = activation
    member _.State = { Weights = weights; Biases = biases }

    member _.Forward(x: float[,]) =
        let z = dot x weights |> addRowVector <| biases
        let a = activation.Forward z
        cache <- Some { Input = x; LinearOutput = z; ActivatedOutput = a }
        a

    member this.Backward(gradOutput: float[,]) =
        let c =
            match cache with
            | Some c -> c
            | None -> failwith "Forward pass must be called before Backward."

        let activationGrad = activation.DerivativeFromOutput c.ActivatedOutput
        let delta = hadamard gradOutput activationGrad
        let batchSize = float (rows c.Input)

        let weightGradients =
            dot (transpose c.Input) delta |> scalarMul (1.0 / batchSize)

        let biasGradients =
            meanColumns delta

        let gradInput =
            dot delta (transpose weights)

        gradInput, { WeightGradients = weightGradients; BiasGradients = biasGradients }

type Layer =
    | Dense of DenseLayer

[<RequireQualifiedAccess>]
module Layers =
    let dense inputSize outputSize activation =
        Dense (DenseLayer(inputSize, outputSize, activation))
