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

package org.apache.seatunnel.api.sink;

/** @deprecated instead by {@link TablePlaceholder} todo remove this class */
@Deprecated
public final class SinkReplaceNameConstant {

    public static final String REPLACE_TABLE_NAME_KEY = "${table_name}";

    public static final String REPLACE_SCHEMA_NAME_KEY = "${schema_name}";

    public static final String REPLACE_DATABASE_NAME_KEY = "${database_name}";
}
