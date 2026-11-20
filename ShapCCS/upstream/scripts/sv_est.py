from args import args
from scripts.init import global_model, client_loaders, validate_loader
from coreset_selection.utils.sv_utils import fed_cssv, fed_cgsv_sample, fed_cgsv, fed_gtg, fed_nsv

if __name__ == '__main__':
    if args.option == "fed_cgsv":
        fed_cgsv(global_model, client_loaders)
    elif args.option == "fed_cgsv_sample":
        fed_cgsv_sample(global_model, client_loaders)
    elif args.option == "fed_cssv":
        fed_cssv(global_model, client_loaders)
    elif args.option == "fed_gtgsv":
        fed_gtg(global_model, client_loaders, validate_loader)
    elif args.option == "fed_nsv":
        fed_nsv(global_model, client_loaders, validate_loader)
    else:
        exit(1)

