open NeuroSharp
open NeuroSharp.Activations
open NeuroSharp.Layers
open NeuroSharp.Losses
open NeuroSharp.Optimizers
open Examples.ExampleData

let printHeader title =
    printfn ""
    printfn "=== %s ===" title

let runXor () =
    printHeader "XOR classification"
    let ds = xor ()
    let model =
        sequential [
            dense 2 8 Activations.tanh
            dense 8 1 Activations.sigmoid
        ]

    train model ds 3000 4 Losses.binaryCrossEntropy (Optimizers.momentum 0.5 0.9) None (verbose = false)
    |> ignore

    let preds = predict model ds.Features
    for r in 0 .. Array2D.length1 preds - 1 do
        printfn $"input=({ds.Features[r,0]:F0}, {ds.Features[r,1]:F0}) -> pred={preds[r,0]:F4}"

let runSineRegression () =
    printHeader "Sine regression"
    let ds = sineRegression 200 |> Dataset.normalizeColumns
    let trainDs, valDs = Dataset.split 0.8 ds
    let model =
        sequential [
            dense 1 16 Activations.tanh
            dense 16 16 Activations.tanh
            dense 16 1 Activations.linear
        ]

    train model trainDs 500 16 Losses.mse (Optimizers.clip 1.0 (Optimizers.sgd 0.05)) None (validationData = valDs, verbose = false)
    |> ignore

    let preds = predict model valDs.Features
    printfn $"validation mse = {Losses.mse.Value valDs.Targets preds:F4}"

let runIrisLike () =
    printHeader "Iris-like 3-class classification"
    let ds = irisLike ()
    let trainDs, valDs = Dataset.split 0.8 ds

    let model =
        sequential [
            dense 4 12 Activations.relu
            dense 12 12 Activations.relu
            dense 12 3 Activations.linear
        ]

    train
        model
        trainDs
        250
        16
        Losses.categoricalCrossEntropy
        (Optimizers.clip 5.0 (Optimizers.momentum 0.05 0.9))
        (Some Metrics.accuracy)
        (validationData = valDs, verbose = false)
    |> ignore

    let preds = predict model valDs.Features
    printfn $"validation accuracy = {Metrics.accuracy valDs.Targets preds:F4}"

[<EntryPoint>]
let main _ =
    runXor ()
    runSineRegression ()
    runIrisLike ()
    0
