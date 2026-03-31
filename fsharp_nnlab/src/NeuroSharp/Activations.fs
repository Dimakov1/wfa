namespace NeuroSharp

open NeuroSharp.Math

type Activation =
    {
        Name: string
        Forward: float[,] -> float[,]
        DerivativeFromOutput: float[,] -> float[,]
    }

[<RequireQualifiedAccess>]
module Activations =
    let relu =
        {
            Name = "ReLU"
            Forward = mapMatrix (fun x -> max 0.0 x)
            DerivativeFromOutput = mapMatrix (fun y -> if y > 0.0 then 1.0 else 0.0)
        }

    let sigmoid =
        {
            Name = "Sigmoid"
            Forward = mapMatrix (fun x -> 1.0 / (1.0 + exp -x))
            DerivativeFromOutput = mapMatrix (fun y -> y * (1.0 - y))
        }

    let tanh =
        {
            Name = "Tanh"
            Forward = mapMatrix System.Math.Tanh
            DerivativeFromOutput = mapMatrix (fun y -> 1.0 - y * y)
        }

    let linear =
        {
            Name = "Linear"
            Forward = id
            DerivativeFromOutput = mapMatrix (fun _ -> 1.0)
        }
