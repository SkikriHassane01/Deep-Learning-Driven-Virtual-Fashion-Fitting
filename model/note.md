here is the `human_parser_model.pth` and the `model_config.json` configuration that will lock like this:

```
{
  "model": {
    "architecture": "HumanParsingNet",
    "num_classes": 18,
    "input_size": [
      512,
      512
    ],
    "model_path": "content/best_model.pth"
  },
  "classes": {
    "names": [
      "Background",
      "Hat",
      "Hair",
      "Sunglasses",
      "Upper-clothes",
      "Skirt",
      "Pants",
      "Dress",
      "Belt",
      "Left-shoe",
      "Right-shoe",
      "Face",
      "Left-leg",
      "Right-leg",
      "Left-arm",
      "Right-arm",
      "Bag",
      "Scarf"
    ],
    "colors": [
      [
        0,
        0,
        0
      ],
      [
        128,
        0,
        0
      ],
      [
        255,
        0,
        0
      ],
      [
        0,
        85,
        0
      ],
      [
        170,
        0,
        51
      ],
      [
        255,
        85,
        0
      ],
      [
        0,
        0,
        85
      ],
      [
        0,
        119,
        221
      ],
      [
        85,
        85,
        0
      ],
      [
        0,
        85,
        85
      ],
      [
        85,
        51,
        0
      ],
      [
        52,
        86,
        128
      ],
      [
        0,
        128,
        0
      ],
      [
        0,
        0,
        255
      ],
      [
        51,
        170,
        221
      ],
      [
        0,
        255,
        255
      ],
      [
        85,
        255,
        170
      ],
      [
        170,
        255,
        85
      ]
    ]
  },
  "preprocessing": {
    "mean": [
      0.485,
      0.456,
      0.406
    ],
    "std": [
      0.229,
      0.224,
      0.225
    ],
    "size": [
      512,
      512
    ]
  },
  "training_info": {
    "dataset": "mattmdjaga/human_parsing_dataset",
    "batch_size": 10,
    "epochs": 10,
    "best_miou": 0.7415336813696529,
    "timestamp": "2025-09-02 15:14:31"
  },
  "evaluation_results": {
    "overall_miou": 0.7415852535575893,
    "per_class_ious": [
      0.9811062520905879,
      0.9456316899296705,
      0.7709287223024677,
      0.8711053147996738,
      0.8343150333781936,
      0.9167979189339842,
      0.8924662524719328,
      0.9185587691554281,
      0.8865493611162814,
      0.3892579494075222,
      0.3525110271731147,
      0.8099577110606371,
      0.6162968909177479,
      0.5071550853379732,
      0.43313370920745914,
      0.41540483173726356,
      0.841447566812707,
      0.9659104782039631
    ],
    "best_classes": [
      [
        "Background",
        0.9811062520905879
      ],
      [
        "Scarf",
        0.9659104782039631
      ],
      [
        "Hat",
        0.9456316899296705
      ]
    ],
    "worst_classes": [
      [
        "Right-shoe",
        0.3525110271731147
      ],
      [
        "Left-shoe",
        0.3892579494075222
      ],
      [
        "Right-arm",
        0.41540483173726356
      ]
    ],
    "class_pixel_counts": [
      715216168.0,
      2045145.0,
      21006729.0,
      628646.0,
      60055934.0,
      16225644.0,
      28221707.0,
      23432043.0,
      520302.0,
      4499734.0,
      4479184.0,
      11058320.0,
      8070187.0,
      7945257.0,
      6695747.0,
      6789239.0,
      9843993.0,
      1780069.0
    ]
  }
}
```