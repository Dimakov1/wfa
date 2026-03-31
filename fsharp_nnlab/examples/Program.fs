open NeuroSharp
open Examples

let runXor () =
    printfn "=== XOR ==="

    let x =
        array2D [
            [0.0; 0.0]
            [0.0; 1.0]
            [1.0; 0.0]
            [1.0; 1.0]
        ]

    let y =
        array2D [
            [0.0]
            [1.0]
            [1.0]
            [0.0]
        ]

    let dataset = Dataset.create x y

    let model =
        sequential [
            Layers.dense 2 8 Activations.tanh
            Layers.dense 8 1 Activations.sigmoid
        ]

    train
        model
        dataset
        2000
        4
        Losses.binaryCrossEntropy
        (Optimizers.sgd 0.1)
        None
        None
        (Some true)
    |> ignore

    let pred = predict model x
    printfn "Predictions:"
    printfn "%A" pred


let runSineRegression () =
    printfn "\n=== Sine regression ==="

    let xs =
        [|
            for i in 0 .. 99 ->
                let x = float i / 10.0
                [| x |]
        |]

    let ys =
        [|
            for i in 0 .. 99 ->
                let x = float i / 10.0
                [| sin x |]
        |]

    let x = array2D xs
    let y = array2D ys

    let dataset = Dataset.create x y

    let model =
        sequential [
            Layers.dense 1 16 Activations.tanh
            Layers.dense 16 16 Activations.tanh
            Layers.dense 16 1 Activations.linear
        ]

    train
        model
        dataset
        1000
        8
        Losses.mse
        (Optimizers.momentum 0.01 0.9)
        None
        None
        (Some true)
    |> ignore

    let pred = predict model x
    printfn "First predictions:"
    for i in 0 .. 9 do
        printfn "x=%.2f y=%.4f pred=%.4f" x[i,0] y[i,0] pred[i,0]


let runIrisLike () =
    printfn "\n=== Iris-like classification ==="

    let dataset = ExampleData.irisLike()

    let model =
        sequential [
            Layers.dense 4 12 Activations.relu
            Layers.dense 12 8 Activations.relu
            Layers.dense 8 3 Activations.sigmoid
        ]

    let optimizer =
        Optimizers.clip 1.0 (Optimizers.momentum 0.01 0.9)

    train
        model
        dataset
        300
        16
        Losses.categoricalCrossEntropy
        optimizer
        None
        None
        (Some true)
    |> ignore

    printfn "Iris-like example finished"


[<EntryPoint>]
let main _ =
    runXor ()
    runSineRegression ()
    runIrisLike ()
    0