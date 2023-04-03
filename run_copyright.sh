

# 1. first find all source files
# 2. remove copywrite
find . -regex '.*\.\(cpp\|hpp\|cu\|c\|h\)' \
  -not -path "*/thirdparty/*" -not -path "*/cmake-build-debug/*" -not -path "*/.idea/*" \
  -exec sed -i '1,/\*\//{ /\/\*/,/\*\//d }' {} \;

# 3. Create copytext file
COPYHEADER="/*
 * ov_plane: Monocular Visual-Inertial Odometry with Planar Regularities
 * Copyright (C) 2022-2023 Chuchu Chen
 * Copyright (C) 2022-2023 Patrick Geneva
 * Copyright (C) 2022-2023 Guoquan Huang
 *
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

"
echo "$COPYHEADER" > /tmp/out_copy


# 4. now append the new header to the files!
find . -regex '.*\.\(cpp\|hpp\|cu\|c\|h\)' \
  -not -path "*/thirdparty/*" -not -path "*/cmake-build-debug/*" -not -path "*/.idea/*" \
  -print0 |
  while IFS= read -r -d '' file; do
    echo $file;
    cat /tmp/out_copy $file > /tmp/out && mv /tmp/out $file
  done


