import argparse

from copy import deepcopy

from frag_gt.src.fragstore import MemoryFragStore, fragstore_factory


def filter_fragstore(old_fragstore, count_limit=1):
    new_store = deepcopy(old_fragstore)

    for gene_type in old_fragstore["gene_types"]:
        for hap, hap_vals in old_fragstore["gene_types"][gene_type]["haplotypes"].items():
            new_gene_frags = {}
            for smi, count in hap_vals["gene_frags"].items():
                if count["count"] >= count_limit:
                    new_gene_frags[smi] = count
            if len(new_gene_frags):
                new_store["gene_types"][gene_type]["haplotypes"][hap]["gene_frags"] = new_gene_frags
            else:
                try:
                    del new_store["gene_types"][gene_type]["haplotypes"][hap]
                except KeyError:
                    pass

        new_n = len(new_store["gene_types"][gene_type]["haplotypes"])
        old_n = len(old_fragstore["gene_types"][gene_type]["haplotypes"])

        gt_deleted = ""
        if not len(new_store["gene_types"][gene_type]["haplotypes"]):
            gt_deleted = "(GeneType deleted)"
            del new_store["gene_types"][gene_type]

        print(f"filter gene_type: {gene_type} ({old_n}->{new_n}) {gt_deleted}")

    print("N gene_types before: ", len(old_fragstore["gene_types"]))
    print("N gene_types after: ", len(new_store["gene_types"]))
    return new_store


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragstore_path", type=str, help="path to .pkl for 'in_memory' fragment store")
    parser.add_argument("--frequency_cutoff", type=int, help="number of occurrences of fragment required to survive")
    return parser


if __name__ == "__main__":
    # parse
    args = get_arg_parser().parse_args()

    # load
    fragstore: MemoryFragStore = fragstore_factory("in_memory", args.fragstore_path)
    fragstore.load()

    # extract store from fragstore object and filter
    filtered_store = filter_fragstore(fragstore.store, args.frequency_cutoff)

    # update fragstore in memory
    fragstore.store = filtered_store

    # save new fragstore to file
    new_filename = f"{args.fragstore_path.rsplit('.', 1)[0]}_filter{args.frequency_cutoff}.pkl"
    fragstore.save(new_filename)
    print(f"Saved filtered fragstore to {new_filename}")
