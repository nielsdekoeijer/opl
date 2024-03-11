set -e
cargo build
cp ./target/debug/libopl.so opl.so
pytest test.py
