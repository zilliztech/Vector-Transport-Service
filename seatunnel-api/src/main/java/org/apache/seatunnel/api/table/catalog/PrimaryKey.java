/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.seatunnel.api.table.catalog;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

@Data
@AllArgsConstructor
public class PrimaryKey implements Serializable {
    private static final long serialVersionUID = 1L;

    // This field is not used now
    private final String primaryKey;

    private final List<String> columnNames;

    private Boolean enableAutoId;

    public PrimaryKey(String primaryKey, List<String> columnNames) {
        this.primaryKey = primaryKey;
        this.columnNames = columnNames;
        this.enableAutoId = null;
    }

    public static boolean isPrimaryKeyField(PrimaryKey primaryKey, String fieldName) {
        if (primaryKey == null || primaryKey.getColumnNames() == null) {
            return false;
        }
        return primaryKey.getColumnNames().contains(fieldName);
    }

    public static PrimaryKey of(String primaryKey, List<String> columnNames, Boolean autoId) {
        return new PrimaryKey(primaryKey, columnNames, autoId);
    }

    public static PrimaryKey of(String primaryKey, List<String> columnNames) {
        return new PrimaryKey(primaryKey, columnNames);
    }

    public PrimaryKey copy() {
        return PrimaryKey.of(primaryKey, new ArrayList<>(columnNames));
    }
}
