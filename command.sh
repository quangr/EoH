#!/bin/bash

# Ensure GNU Parallel is installed (e.g., sudo apt-get install parallel)

parallel -j 3 << EOF
/bin/python3 /home/mcgubio/EoH/test.py --exp_output_path output/10_0 --eva_timeout 10
/bin/python3 /home/mcgubio/EoH/test.py --exp_output_path output/50_0 --eva_timeout 50
EOF

# parallel -j 3 << EOF
# /bin/python3 /home/mcgubio/EoH/test.py --exp_output_path output/10_0 --eva_timeout 10
# /bin/python3 /home/mcgubio/EoH/test.py --exp_output_path output/10_1 --eva_timeout 10
# /bin/python3 /home/mcgubio/EoH/test.py --exp_output_path output/10_2 --eva_timeout 10
# /bin/python3 /home/mcgubio/EoH/test.py --exp_output_path output/50_0 --eva_timeout 50
# /bin/python3 /home/mcgubio/EoH/test.py --exp_output_path output/50_1 --eva_timeout 50
# /bin/python3 /home/mcgubio/EoH/test.py --exp_output_path output/50_2 --eva_timeout 50
# /bin/python3 /home/mcgubio/EoH/test.py --exp_output_path output/100_0 --eva_timeout 100
# /bin/python3 /home/mcgubio/EoH/test.py --exp_output_path output/100_1 --eva_timeout 100
# /bin/python3 /home/mcgubio/EoH/test.py --exp_output_path output/100_2 --eva_timeout 100
# /bin/python3 /home/mcgubio/EoH/test.py --exp_output_path output/500_0 --eva_timeout 500
# /bin/python3 /home/mcgubio/EoH/test.py --exp_output_path output/500_1 --eva_timeout 500
# /bin/python3 /home/mcgubio/EoH/test.py --exp_output_path output/500_2 --eva_timeout 500
# EOF