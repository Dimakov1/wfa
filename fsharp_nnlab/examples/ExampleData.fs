namespace Examples

open NeuroSharp
open NeuroSharp.Math

[<RequireQualifiedAccess>]
module ExampleData =
    let xor () =
        let x =
            array2D [|
                [| 0.0; 0.0 |]
                [| 0.0; 1.0 |]
                [| 1.0; 0.0 |]
                [| 1.0; 1.0 |]
            |]

        let y =
            array2D [|
                [| 0.0 |]
                [| 1.0 |]
                [| 1.0 |]
                [| 0.0 |]
            |]

        Dataset.create x y

    let sineRegression samples =
        let x =
            Array.init samples (fun i ->
                let value = -3.14 + 6.28 * float i / float (samples - 1)
                [| value |])

        let y =
            x |> Array.map (fun row -> [| sin row[0] |])

        Dataset.create (Tensor.fromJagged x) (Tensor.fromJagged y)

    let irisLike () =
        let centers =
            [|
                ([| 5.0; 3.5; 1.4; 0.2 |], 0)
                ([| 6.0; 2.9; 4.5; 1.4 |], 1)
                ([| 6.5; 3.0; 5.5; 2.0 |], 2)
            |]

        let samples =
            [|
                for classCenter, label in centers do
                    for _ in 1 .. 50 do
                        let noisy =
                            classCenter
                            |> Array.map (fun v -> v + (random.NextDouble() * 2.0 - 1.0) * 0.25)
                        yield noisy, label
            |]

        let x = samples |> Array.map fst |> Tensor.fromJagged
        let y = samples |> Array.map snd |> Tensor.ofLabels 3
        Dataset.create x y |> Dataset.normalizeColumns
