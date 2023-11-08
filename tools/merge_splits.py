import json

test_direct = json.load(open("/home/quang/workspace/lavin-original/data/dhpr_annotations/v1107_meta_anno_test_direct.json"))
test_indirect = json.load(open("/home/quang/workspace/lavin-original/data/dhpr_annotations/v1107_meta_anno_test_indirect.json"))

all_test = {}
for k, v in test_direct.items():
    all_test[k] = v

for k, v in test_indirect.items():
    all_test[k] = v

test_direct_keys = list(test_direct.keys())
test_indirect_keys = list(test_indirect.keys())

new_val_keys = test_direct_keys[:500] + test_indirect_keys[:500]
new_test_keys = test_direct_keys[500:1000] + test_indirect_keys[500:1000]
new_val = {k: all_test[k] for k in new_val_keys}
new_test = {k: all_test[k] for k in new_test_keys}

with open("/home/quang/workspace/lavin-original/data/dhpr_annotations/v1108_meta_anno_test.json", 'w') as f:
    json.dump(new_test, f, indent=4)

with open("/home/quang/workspace/lavin-original/data/dhpr_annotations/v1108_meta_anno_val.json", 'w') as f:
    json.dump(new_val, f, indent=4)

val_direct = json.load(open("/home/quang/workspace/lavin-original/data/dhpr_annotations/v1107_meta_anno_val_direct.json"))
val_indirect = json.load(open("/home/quang/workspace/lavin-original/data/dhpr_annotations/v1107_meta_anno_val_indirect.json"))
train = json.load(open("/home/quang/workspace/lavin-original/data/dhpr_annotations/v1107_meta_anno_train.json"))

new_train = {}
for k, v in train.items():
    new_train[k] = v

for k, v in val_direct.items():
    new_train[k] = v

for k, v in val_indirect.items():
    new_train[k] = v

print("Total train:", len(new_train))

with open("/home/quang/workspace/lavin-original/data/dhpr_annotations/v1108_meta_anno_train.json", 'w') as f:
    json.dump(new_train, f, indent=4)

# Re-Check
train = json.load(open("/home/quang/workspace/lavin-original/data/dhpr_annotations/v1108_meta_anno_train.json"))
test = json.load(open("/home/quang/workspace/lavin-original/data/dhpr_annotations/v1108_meta_anno_test.json"))
val = json.load(open("/home/quang/workspace/lavin-original/data/dhpr_annotations/v1108_meta_anno_val.json"))
print(len(train), len(test), len(val))
